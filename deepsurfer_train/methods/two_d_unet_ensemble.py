"""Implementation of a 2D UNet ensemble method."""
from typing import Any

import einops
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typeguard import typechecked

from monai.losses.dice import DiceCELoss
from monai.metrics.meandice import DiceMetric
from monai.networks.nets.unet import UNet


from deepsurfer_train.enums import SpatialFormat
from deepsurfer_train.methods.base import MethodProtocol
from deepsurfer_train.preprocess.dataset import DeepsurferSegmentationDataset
from deepsurfer_train.visualization.tensorboard import (
    write_batch_multiplanar_tensorboard,
)


@typechecked
def reformat_image(
    t: torch.Tensor,
    spatial_format: SpatialFormat,
    channel_stack: int = 1,
) -> torch.Tensor:
    """Move a tensor into an alternative spatial format.

    Given a tensor with a 3D spatial format (B, C, H, W, D) in PLI orientation,
    move the image into the desired spatial format with individual 2D slices
    stacked down the batch axis.

    Parameters
    ----------
    t: torch.Tensor
        Tensor in 3D (B, C, H, W, D) format and PLI spatial orientation.
    spatial_format: deepsurfer_train.enums.SpatialFormat
        Spatial format of the desired output tensor.
    channel_stack: int
        Stack this many consecutive down the channel axis for the fastsurfer-
        style 2D network. Has no effect if the spatial_format is not 2D.

    Returns
    ------
    torch.Tensor:
        Tensor in the requested output format.

    """
    if t.ndim != 5:
        raise ValueError("Input must be a 5D tensor")
    if channel_stack % 2 != 1:
        raise ValueError("channel_stack must be an odd integer")
    if channel_stack == 1:
        # More efficient in case of a single output channel
        if spatial_format == SpatialFormat.TWO_D_AXIAL:
            t = einops.rearrange(t, "b c p l i -> (b i) c p l")
        if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
            t = einops.rearrange(t, "b c p l i -> (b l) c p i")
        if spatial_format == SpatialFormat.TWO_D_CORONAL:
            t = einops.rearrange(t, "b c p l i -> (b p) c l i")
    else:
        if t.shape[1] != 1:
            raise ValueError("Can only deal with single channel tensors")
        # NB in pytorch 2.0 we could use Tensor.unroll()
        pad_slices = channel_stack // 2
        if spatial_format == SpatialFormat.TWO_D_AXIAL:
            d = t.shape[4]
            pad = [pad_slices, pad_slices]
            t = torch.nn.functional.pad(t, pad)
            output_slices = []
            for s in range(d):
                window = t[:, :, :, :, s : s + channel_stack]
                window = einops.rearrange(window, "b 1 p l i -> b i p l")
                output_slices.append(window)
            t = torch.cat(output_slices, dim=0)
        if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
            d = t.shape[3]
            pad = [0, 0, pad_slices, pad_slices]
            t = torch.nn.functional.pad(t, pad)
            output_slices = []
            for s in range(d):
                window = t[:, :, :, s : s + channel_stack, :]
                window = einops.rearrange(window, "b 1 p l i -> b l p i")
                output_slices.append(window)
            t = torch.cat(output_slices, dim=0)
        if spatial_format == SpatialFormat.TWO_D_CORONAL:
            d = t.shape[2]
            pad = [0, 0, 0, 0, pad_slices, pad_slices]
            t = torch.nn.functional.pad(t, pad)
            output_slices = []
            for s in range(d):
                window = t[:, :, s : s + channel_stack, :, :]
                window = einops.rearrange(window, "b 1 p l i -> b p l i")
                output_slices.append(window)
            t = torch.cat(output_slices, dim=0)

    return t


@typechecked
def inverse_reformat_image(
    t: torch.Tensor, spatial_format: SpatialFormat, batch_size: int
) -> torch.Tensor:
    """Undo the operation performed in reformat_image.

    Given a tensor with a shape output by the reformat_image function, undo it
    and return to (B, C, H, W, D) format in PLI orientation.

    Parameters
    ----------
    t: torch.Tensor
        Tensor in either 2D or 3D format as output by reformat_image.
    spatial_format: deepsurfer_train.enums.SpatialFormat
        Spatial format of the input tensor.
    batch_size: int
        Batch size as it was before the forward transform.

    Returns
    ------
    torch.Tensor:
        Tensor in the requested output format.

    """
    if spatial_format == SpatialFormat.TWO_D_AXIAL:
        return einops.rearrange(t, "(b i) c p l -> b c p l i", b=batch_size)
    if spatial_format == SpatialFormat.TWO_D_CORONAL:
        return einops.rearrange(t, "(b p) c l i -> b c p l i", b=batch_size)
    if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
        return einops.rearrange(t, "(b l) c p i -> b c p l i", b=batch_size)

    return t


@typechecked
def aggregate_views(
    predictions: dict[SpatialFormat, torch.Tensor],
    sagittal_unmerging_indices: list[int],
) -> torch.Tensor:
    """Aggregrate 2D predictions from the three planes.

    This follows the "fastsurfer" implementation.

    Henschel et al 2020. "FastSurfer - A fast and accurate deep learning based
    neuroimaging pipeline"

    Parameters
    ----------
    predictions: torch.Tensor
        Prediction from the each model. Each key is a spatial format and each
        value is a tensor in PLI 3D format (B x C x H x W x D) reformatted from
        the output of a 2D model. The tensor contains the raw logit outputs
        before softmax or thresholding. Note that it is note required to include
        all spatial formats.
    sagittal_unmerging_indices: list[int]
        List of indices used to invert the merging of the lateral classes in
        the sagittal model's prediction. This is a list of channel indices:
        item i with value v specifies that the ith channel of the "unmerged"
        array (with separate left and right classes) is drawn from channel v of
        the "merged" sagittal array (with combined left/right classes).

    Returns
    -------
    torch.Tensor:
        Merged prediction tensor. It has the same shape as the input tensors
        and the values are pseudo-probabilities (post softmax).

    """
    t = list(predictions.values())[0]
    c = len(sagittal_unmerging_indices)
    b, _, h, w, d = t.shape
    aggregated = torch.zeros((b, c, h, w, d))
    weights = 0.0
    for spatial_format, t in predictions.items():
        if spatial_format == SpatialFormat.THREE_D:
            raise ValueError(
                "THREE_D prediction is not a valid input to the aggregation method."
            )
        elif spatial_format == SpatialFormat.TWO_D_SAGITTAL:
            unmerged = t[:, sagittal_unmerging_indices, :, :, :]
            aggregated += 0.5 * torch.softmax(unmerged, dim=1)
            weights += 0.5
        else:
            aggregated += torch.softmax(t, dim=1)
            weights += 1.0

    aggregated /= weights

    return aggregated


@typechecked
class UNet2DEnsembleMethod(MethodProtocol):
    """A method using an ensemble of 2D UNets segmentation model."""

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
        self.labels = {}

        # One model in each of the three planes
        formats = [SpatialFormat(f) for f in method_params["spatial_formats"]]
        self.channel_stack = int(method_params["channel_stack"])

        # Set up optimizer and scheduler
        optimizer_cls = getattr(torch.optim, model_config["optimizer"])
        scheduler_cls = getattr(torch.optim.lr_scheduler, model_config["scheduler"])
        self.schedulers = {}
        self.optimizers = {}

        # Sagittal model used lateral merged labels, others use
        # usual set of labels
        n_labels_mapping = {
            SpatialFormat.TWO_D_SAGITTAL: train_dataset.n_total_merged_labels,
            SpatialFormat.TWO_D_AXIAL: train_dataset.n_total_labels,
            SpatialFormat.TWO_D_CORONAL: train_dataset.n_total_labels,
        }
        label_mapping = {
            SpatialFormat.TWO_D_SAGITTAL: train_dataset.merged_labels,
            SpatialFormat.TWO_D_AXIAL: train_dataset.labels,
            SpatialFormat.TWO_D_CORONAL: train_dataset.labels,
        }
        self.out_channels = {}
        self.models = {}
        self.loss_fns = {}
        self.dice_metric = {}
        self.per_class_dice_metrics = {}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        for spatial_format in formats:
            self.out_channels[spatial_format] = n_labels_mapping[spatial_format]
            self.models[spatial_format] = UNet(
                spatial_dims=2,
                in_channels=self.channel_stack,
                out_channels=self.out_channels[spatial_format],
                **model_config["unet_params"],
            )
            self.models[spatial_format].cuda()

            self.labels[spatial_format] = label_mapping[spatial_format]

            if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
                class_weights = train_dataset.merged_weights
            else:
                class_weights = train_dataset.weights
            self.loss_fns[spatial_format] = DiceCELoss(
                include_background=False,
                softmax=True,
                squared_pred=False,
                weight=torch.tensor(class_weights).cuda(),
            )
            self.optimizers[spatial_format] = optimizer_cls(
                self.models[spatial_format].parameters(),
                **model_config["optimizer_params"],
            )
            self.schedulers[spatial_format] = scheduler_cls(
                self.optimizers[spatial_format],
                **model_config["scheduler_params"],
            )
            self.dice_metric[spatial_format] = DiceMetric(
                include_background=False,
                num_classes=len(self.labels[spatial_format]),
            )
            self.per_class_dice_metrics[spatial_format] = {
                c: DiceMetric(include_background=False, num_classes=1)
                for c in self.labels[spatial_format]
            }

    def train_begin(self) -> None:
        """Callback called at the beginning of a training epoch.

        This should do things like move models to train mode and reset training
        statistics.

        """
        for model in self.models.values():
            model.train()
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        self.batch_losses: dict[SpatialFormat, list[float]] = {
            spatial_format: [] for spatial_format in self.models.keys()
        }

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
        # Loop through and run update steps for all of the models in the ensemble
        losses = []
        for spatial_format in self.models.keys():
            model = self.models[spatial_format]
            optimizer = self.optimizers[spatial_format]

            input_image = reformat_image(
                batch[self.train_dataset.image_key],
                spatial_format,
                channel_stack=self.channel_stack,
            )
            if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
                mask_key = self.train_dataset.merged_mask_key
            else:
                mask_key = self.train_dataset.mask_key
            mask_image = reformat_image(
                batch[mask_key],
                spatial_format,
                channel_stack=1,
            )

            prediction = model(input_image)
            loss = self.loss_fns[spatial_format](prediction, mask_image)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_float = loss.detach().cpu().item()
            self.batch_losses[spatial_format].append(loss_float)
            losses.append(loss_float)

        return sum(losses)

    def train_end(self, e: int) -> None:
        """Callback for the end of the training process.

        Generally used for writing out additional epoch level statistics.

        """
        for spatial_format in self.models.keys():
            epoch_loss = np.mean(self.batch_losses[spatial_format])
            self.writer.add_scalar(f"loss/train_{spatial_format.name}", epoch_loss, e)
            print(f"Epoch {e}, train loss ({spatial_format.name}) {epoch_loss:.4f}")

    def val_begin(self, e: int) -> None:
        """Callback called at the beginning of a val epoch.

        This should do things like move models to eval mode and reset val statistics.

        Parameters
        ----------
        e: int
            Epoch number.

        """
        for model in self.models.values():
            model.eval()
        self.val_batch_losses: dict[SpatialFormat, list[float]] = {
            spatial_format: [] for spatial_format in self.models.keys()
        }

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
        losses = []
        predictions_per_format = {}
        gt_labelmaps = batch[self.val_dataset.mask_key].argmax(  # type: ignore
            axis=1, keepdim=True
        )

        for spatial_format in self.models.keys():
            model = self.models[spatial_format]

            input_image = reformat_image(
                batch[self.val_dataset.image_key],  # type: ignore
                spatial_format,
                channel_stack=self.channel_stack,
            )
            if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
                mask_key = self.val_dataset.merged_mask_key
            else:
                mask_key = self.val_dataset.mask_key
            mask_image = reformat_image(
                batch[mask_key],  # type: ignore
                spatial_format,
                channel_stack=1,
            )

            prediction = model(input_image)

            loss = self.loss_fns[spatial_format](prediction, mask_image)

            loss_float = loss.detach().cpu().item()
            self.val_batch_losses[spatial_format].append(loss_float)
            losses.append(loss_float)

            # Move prediction back to 3D format
            batch_size = input_image.shape[0]
            prediction = inverse_reformat_image(
                prediction.detach(),
                spatial_format,
                batch_size,
            )
            predictions_per_format[spatial_format] = prediction

            pred_labelmaps = prediction.argmax(dim=1, keepdim=True)

            # Calculate metrics in 3D
            self.dice_metric[spatial_format](pred_labelmaps, gt_labelmaps)
            for c, label in enumerate(self.labels[spatial_format], 1):
                self.per_class_dice_metrics[spatial_format][label](
                    pred_labelmaps == c,
                    gt_labelmaps == c,
                )

            write_batch_multiplanar_tensorboard(
                writer=self.writer,
                tag_group_prefix="val",
                tag_prefix=f"prediction_{spatial_format.name}",
                images=batch[self.val_dataset.image_key],  # type: ignore
                labelmaps=pred_labelmaps,
                subject_ids=batch["subject_id"],  # type: ignore
                num_classes=self.out_channels[spatial_format],
                step=e,
            )

        total_loss = sum(losses)
        if len(self.models) > 1:
            # Metrics on the aggregated prediction
            aggregated_prediction = aggregate_views(
                predictions=predictions_per_format,
                sagittal_unmerging_indices=self.val_dataset.get_unmerging_indices(),
            )
            aggregated_labelmaps = aggregated_prediction.argmax(dim=1, keepdim=True)
            return total_loss, aggregated_labelmaps
        else:
            return total_loss, list(predictions_per_format.values())[0]

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
        for spatial_format in self.models.keys():
            epoch_loss = np.mean(self.val_batch_losses[spatial_format])
            self.writer.add_scalar("loss/val", epoch_loss, e)
            self.writer.add_scalar(
                f"dice_metric_3d/val_{spatial_format.name}",
                self.dice_metric[spatial_format].aggregate(),
                e,
            )
            self.dice_metric[spatial_format].reset()

            per_class_averages = []
            for label in self.labels[spatial_format]:
                val = (
                    self.per_class_dice_metrics[spatial_format][label]
                    .aggregate()
                    .cpu()  # type: ignore
                    .item()
                )
                self.writer.add_scalar(
                    f"per_class_dice_metric_3d/{spatial_format.name}_{label}",
                    val,
                    e,
                )
                self.per_class_dice_metrics[spatial_format][label].reset()
                per_class_averages.append(val)

            self.writer.add_scalar(
                f"average_dice_metric_3d_{spatial_format.name}",
                np.mean(per_class_averages),
                e,
            )

            print(f"Epoch {e}, val loss ({spatial_format.name}) {epoch_loss:.4f}")
            self.writer.add_scalar(
                f"learning_rate/{spatial_format.name}",
                self.schedulers[spatial_format].optimizer.param_groups[0]["lr"],
                e,
            )
            if isinstance(
                self.schedulers[spatial_format],
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                # special case: need to pass loss to step method
                self.schedulers[spatial_format].step(epoch_loss)
            else:
                self.schedulers[spatial_format].step()

    def get_state_dict(self) -> dict[str, Any]:
        """Get the state dict for all models used by the method.

        Parameters
        ----------
        dict[str, Any]:
            State dict to save.

        """
        return {
            spatial_format.name: model.state_dict()
            for spatial_format, model in self.models.items()
        }
