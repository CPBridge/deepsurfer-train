"""Implementation of a simple 3D Unet segmentation method."""
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typeguard import typechecked

from monai.losses.dice import DiceCELoss
from monai.networks.nets.unet import UNet


from deepsurfer_train.methods.base import MethodProtocol
from deepsurfer_train.preprocess.dataset import DeepsurferSegmentationDataset


@typechecked
class UNet3DMethod(MethodProtocol):
    """A simple method using a 3D Unet segmentation model."""

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
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=train_dataset.n_total_labels,
            **model_config["unet_params"],
        )
        self.loss_fn = DiceCELoss(
            include_background=False,
            softmax=True,
            squared_pred=False,
            weight=torch.tensor(train_dataset.weights).cuda(),
        )
        self.model.cuda()

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
        mask_image = batch[self.train_dataset.mask_key]

        prediction = self.model(input_image)
        loss = self.loss_fn(prediction, mask_image)
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
        input_image = batch[self.val_dataset.image_key]
        mask_image = batch[self.val_dataset.mask_key]
        prediction = self.model(input_image)
        loss = self.loss_fn(prediction, mask_image)
        loss_float = loss.detach().cpu().item()
        pred_labelmaps = prediction.argmax(dim=1, keepdim=True)

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
