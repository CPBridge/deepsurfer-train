"""Implementation of a Bayesian segmentation method."""
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typeguard import typechecked

import monai
from monai.data import MetaTensor
from monai.networks.nets.unet import UNet

from voxelmorph.torch.layers import (
    ResizeTransform,
    SpatialTransformer,
    VecInt,
)
from voxelmorph.torch.losses import Grad
from deepsurfer_train.enums import BrainRegions
from deepsurfer_train.atlas import get_atlas_metatensor
from deepsurfer_train.methods.base import MethodProtocol
from deepsurfer_train.preprocess.dataset import DeepsurferSegmentationDataset


class BayesNet(torch.nn.Module):
    """Implementation of the BayesNet model architecture."""

    def __init__(
        self,
        unet_params: dict[str, Any],
        n_labels: int,
        head_channels: int,
        imsize: tuple[int, int, int],
        int_steps: int = 7,
        int_downsize: int = 2,
    ) -> None:
        """Initializer.

        Parameters
        ----------
        unet_params: dict[str, Any]
            Dictionary of parameters for the backbone UNet.
        n_labels: int
            Number of labels in the atlas (including background).
        head_channels: int
            Number of output channels of UNet.
        imsize: tuple[int, int, int]
            Spatial size of input images (and atlas).
        int_steps: int
            Number of integration steps.
        int_downsize: int
            Number of downsampling steps for calculating warp.

        """
        super().__init__()
        ndims = 3
        n_input_channels = n_labels + 1  # image + atlas channels (1 per label)
        self.backbone = UNet(
            spatial_dims=ndims,
            in_channels=n_input_channels,  # image and atlas
            out_channels=head_channels,
            **unet_params,
        )

        # Two additional layers for the likelhood prediction
        self.mu_head = torch.nn.Conv3d(
            in_channels=head_channels,
            out_channels=n_labels,
            kernel_size=3,
            padding="same",
        )
        self.sigma_head = torch.nn.Conv3d(
            in_channels=head_channels,
            out_channels=n_labels,
            kernel_size=3,
            padding="same",
        )
        # One additional layer for the velocity prediction
        self.velocity_head = torch.nn.Conv3d(
            in_channels=head_channels,
            out_channels=ndims,
            kernel_size=3,
            padding="same",
        )

        # Configure optional resize layers (downsize)
        if int_steps > 0 and int_downsize > 1:
            self._resize = ResizeTransform(int_downsize, ndims)
            self._fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self._resize = None
            self._fullsize = None

        # Configure integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in imsize]
        self._integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # Configure transformer
        self._transformer = SpatialTransformer(imsize)

        if torch.cuda.is_available():
            self._transformer.cuda()
            if self._integrate is not None:
                self._integrate.cuda()
            if self._resize is not None:
                self._resize.cuda()
            if self._fullsize is not None:
                self._fullsize.cuda()

    def forward(
        self, im: torch.Tensor, atlas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the BayesNet, without calculating loss.

        Parameters
        ----------
        im: torch.Tensor
            Input image (B, 1, H, W, D).
        atlas: torch.Tensor
            Atlas image (B, C, H, W, D) where C is number of classes.

        Returns
        -------
        log_posterior: torch.Tensor
            Log posterior image (B, C, H, W, D). Prediction is the argmax of
            this down the channel axis.
        warped_prior: torch.Tensor
            Warped atlas image (B, C, H, W, D).
        flow: torch.Tensor
            Flow used to register the atlas to the image.

        """
        model_input = torch.concat([im, atlas], dim=1)
        features = self.backbone(model_input)

        mu = self.mu_head(features)
        mu = torch.nn.functional.adaptive_avg_pool3d(mu, (1, 1, 1)).squeeze()
        sigma_2 = self.sigma_head(features)
        sigma_2 = torch.nn.functional.elu(sigma_2) + 1.0  # force positive
        sigma_2 = torch.nn.functional.adaptive_avg_pool3d(sigma_2, (1, 1, 1)).squeeze()

        flow = self.velocity_head(features)

        # resize flow for integration
        if self._resize:
            flow = self._resize(flow)

        preint_flow = flow
        # inv_flow = -flow

        # integrate to produce diffeomorphic warp
        if self._integrate is not None:
            flow = self._integrate(flow)
            # inv_flow = self._integrate(inv_flow)

            if self._fullsize:
                flow = self._fullsize(flow)
                # inv_flow = self._fullsize(inv_flow)

        # warp image with flow field
        warped_prior = self._transformer(atlas, flow)
        # warped_im = self._transformer(image, inv_flow)

        nlabels = mu.shape[0]

        # Need to flatten the image to pass to gaussian likelihood function
        # and repeat along channel axis in order to find likelihood under
        # multiple mean/variance combos
        # TODO this will only work for batch size 1 at the moment, needs to
        # be generalized
        repeated_flat_im = im.reshape([1, -1]).expand([nlabels, -1])
        log_likelihood_flat = -torch.nn.functional.gaussian_nll_loss(
            input=repeated_flat_im,
            target=mu[:, None],
            var=sigma_2[:, None],
            reduction="none",
            full=True,
        )

        # Put the log likelihood back in the image shape
        log_likelihood = log_likelihood_flat.view(atlas.shape)

        log_posterior = log_likelihood + torch.log(warped_prior)

        return log_posterior, warped_prior, preint_flow


@typechecked
class BayesNetMethod(MethodProtocol):
    """Bayesnet segmentation method.

    Follows Dalca's method.

    Dalca et al 2019. Unsupervised Deep Learning for Bayesian Brain MRI
    Segmentation.

    """

    def __init__(
        self,
        model_config: dict[str, Any],
        method_params: dict[str, Any],
        writer: SummaryWriter,
        train_dataset: DeepsurferSegmentationDataset,
        val_dataset: DeepsurferSegmentationDataset,
    ) -> None:
        """Initializer.

        Parameters
        ----------
        model_config: dict[str, Any]
            Model configuration parameter dictionary.
        method_params: dict[str, Any]
            Method-specific configuration parameter dictionary.
        writer: SummaryWriter
            Tensorboard writer object to use to record images and scores.
        train_dataset: DeepsurferSegmentationDataset
            The training dataset.
        val_dataset: DeepsurferSegmentationDataset
            The validation dataset.

        """
        self.writer = writer
        self.labels = train_dataset.labels
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = BayesNet(
            unet_params=model_config["unet_params"],
            head_channels=method_params["head_channels"],
            n_labels=len(self.labels) + 1,
            imsize=tuple(self.train_dataset.imsize),
            int_steps=method_params["int_steps"],
            int_downsize=method_params["int_downsize"],
        )
        self.model.cuda()
        self.atlas = self.get_preprocessed_atlas()[None]

        # Flatten the atlas in order to create a reasonably sized input image
        # for the voxelmorph network
        self._grad_loss = Grad("l2", loss_mult=method_params["int_downsize"])
        self._grad_loss_weight = method_params["grad_loss_weight"]

        # Set up optimizer and scheduler
        optimizer_cls = getattr(torch.optim, model_config["optimizer"])
        scheduler_cls = getattr(torch.optim.lr_scheduler, model_config["scheduler"])
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            **model_config["optimizer_params"],
        )
        self.scheduler = scheduler_cls(
            self.optimizer,
            **model_config["scheduler_params"],
        )

    def get_preprocessed_atlas(self) -> MetaTensor:
        """Load and preprocess the atlas.

        Returns
        -------
        monai.data.MetaTensor:
            Loaded atlas as a monai metatensor.

        """
        regions = [BrainRegions[r] for r in self.train_dataset.labels]
        atlas = get_atlas_metatensor(regions).cuda()

        transforms = monai.transforms.Compose(
            [
                monai.transforms.EnsureChannelFirst(),
                monai.transforms.Orientation(axcodes="PLI", lazy=True),
                monai.transforms.ResizeWithPadOrCrop(
                    spatial_size=self.train_dataset.imsize,
                    lazy=True,
                ),
            ],
            lazy=True,
        )
        atlas_out = transforms(atlas)
        return atlas_out

    def train_begin(self) -> None:
        """Callback called at the beginning of a training epoch.

        This should do things like move models to train mode and reset training
        statistics.

        """
        self.model.train()
        self.optimizer.zero_grad()

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """A single training step.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            Batch dictionary as prodcuced by the dataset.

        Returns
        -------
        float:
            Primary loss value that will be written to the tensorboard. If
            further sub-losses are relevant, they may be written to writer in
            this method.

        """
        input_image = batch[self.train_dataset.image_key]

        log_likelihood, _, flow = self.model(input_image, self.atlas)
        nll_loss = -torch.mean(log_likelihood)
        grad_loss = self._grad_loss.loss(None, flow)
        loss = nll_loss + self._grad_loss_weight * grad_loss
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach().cpu().item()

    def train_end(self, e: int) -> None:
        """Callback for the end of the training process.

        Generally used for writing out additional epoch level statistics.

        Parameters
        ----------
        e: int
            Epoch number.

        """
        pass

    def val_begin(self, e: int) -> None:
        """Callback called at the beginning of a val epoch.

        This should do things like move models to eval mode and reset val statistics.

        Parameters
        ----------
        e: int
            Epoch number.

        """
        self.model.eval()

    def val_step(
        self, batch: dict[str, torch.Tensor | str], e: int
    ) -> tuple[float, torch.Tensor]:
        """A single validation step.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            Batch dictionary as prodcuced by the dataset.
        e: int
            Epoch number.

        Returns
        -------
        float:
            The validtion loss on this batch.
        torch.Tensor:
            Predicted mask in labelmap format.

        """
        input_image = batch[self.train_dataset.image_key]

        log_likelihood, _, flow = self.model(input_image, self.atlas)
        nll_loss = -torch.mean(log_likelihood)
        grad_loss = self._grad_loss.loss(None, flow)
        loss = nll_loss + self._grad_loss_weight * grad_loss
        loss_float = loss.detach().cpu().item()
        pred_labelmaps = log_likelihood.argmax(dim=1, keepdim=True)

        return loss_float, pred_labelmaps

    def val_end(self, e: int, val_loss: float) -> None:
        """Callback for the end of the validation process.

        Generally used for writing out additional epoch level statistics and
        updating schedulers.

        Parameters
        ----------
        e: int
            Epoch number.
        val_loss: float
            Validation loss for this epoch (for updating scheduler).

        """
        self.writer.add_scalar(
            "learning_rate",
            self.scheduler.optimizer.param_groups[0]["lr"],
            e,
        )
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # special case: need to pass loss to step method
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def get_state_dict(self) -> dict[str, Any]:
        """Get the state dict for all models used by the method.

        Parameters
        ----------
        dict[str, Any]:
            State dict to save.

        """
        return self.model.state_dict()
