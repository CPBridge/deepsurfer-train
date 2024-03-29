"""Training loop logic."""
from pathlib import Path
from typing import Any

import einops
from monai.metrics.meandice import DiceMetric
from monai.networks.nets.unet import UNet
from monai.losses.dice import DiceCELoss
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typeguard import typechecked


from deepsurfer_train.enums import (
    DatasetPartition,
    SpatialFormat,
    ExperimentType,
)
from deepsurfer_train import locations
from deepsurfer_train.preprocess.dataset import DeepsurferSegmentationDataset
from deepsurfer_train.visualization.tensorboard import (
    write_batch_multiplanar_tensorboard,
)
from deepsurfer_train.models.fastsurfer import aggregate_views


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


def training_loop(
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    output_dir: Path | str,
) -> None:
    """Main training loop logic. Trains a model given config dicts.

    Parameters
    ----------
    data_config: dict[str, Any]
        Configuration parameters for the dataset.
    model_config: dict[str, Any]
        Configuration for the model and training process.
    output_dir: Path | str
        Output directory for weights and logs.

    """
    use_gpu = True
    output_dir = Path(output_dir)

    print("Output dir:", output_dir.resolve())

    # Set up dataset
    train_dataset = DeepsurferSegmentationDataset(
        dataset=data_config["dataset"],
        partition=DatasetPartition.TRAIN,
        processed_version=data_config.get("processed_version", None),
        root_dir=data_config.get("root_dir", locations.project_dataset_dir),
        imsize=data_config["imsize"],
        use_spatial_augmentation=True,
        use_intensity_augmentation=True,
        synth_probability=data_config["synth_probability"],
        excluded_regions=data_config.get("excluded_regions"),
        use_gpu=use_gpu,
    )
    val_dataset = DeepsurferSegmentationDataset(
        dataset=data_config["dataset"],
        partition=DatasetPartition.VALIDATION,
        processed_version=data_config.get("processed_version", None),
        root_dir=data_config.get("root_dir", locations.project_dataset_dir),
        imsize=data_config["imsize"],
        use_spatial_augmentation=False,
        use_intensity_augmentation=False,
        synth_probability=0.0,  # ?
        excluded_regions=data_config.get("excluded_regions"),
        use_gpu=use_gpu,
    )

    # Set up data loader
    batch_size = model_config["batch_size"]
    batches_per_epoch = model_config["batches_per_epoch"]
    samples_per_epoch = batches_per_epoch * batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(
            range(len(train_dataset)),
            replacement=False,
            num_samples=samples_per_epoch,
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
    )

    # Set up model
    experiment_type = ExperimentType(model_config["experiment_type"])
    if experiment_type == ExperimentType.TWO_D_ENSEMBLE:
        # One model in each of the three planes
        formats = [
            SpatialFormat.TWO_D_SAGITTAL,
            SpatialFormat.TWO_D_AXIAL,
            SpatialFormat.TWO_D_CORONAL,
        ]
        channel_stack = int(model_config["channel_stack"])

        # Sagittal model used lateral merged labels, others use
        # usual set of labels
        out_channels = {
            SpatialFormat.TWO_D_SAGITTAL: train_dataset.n_total_merged_labels,
            SpatialFormat.TWO_D_AXIAL: train_dataset.n_total_labels,
            SpatialFormat.TWO_D_CORONAL: train_dataset.n_total_labels,
        }
        labels = {
            SpatialFormat.TWO_D_SAGITTAL: train_dataset.merged_labels,
            SpatialFormat.TWO_D_AXIAL: train_dataset.labels,
            SpatialFormat.TWO_D_CORONAL: train_dataset.labels,
        }
        models = {
            spatial_format: UNet(
                spatial_dims=2,
                in_channels=channel_stack,
                out_channels=out_channels[spatial_format],
                **model_config["unet_params"],
            )
            for spatial_format in formats
        }
        loss_fn = {
            SpatialFormat.TWO_D_SAGITTAL: DiceCELoss(
                include_background=False,
                softmax=True,
                squared_pred=False,
                weight=torch.tensor(train_dataset.merged_weights).cuda(),
            ),
            SpatialFormat.TWO_D_AXIAL: DiceCELoss(
                include_background=False,
                softmax=True,
                squared_pred=False,
                weight=torch.tensor(train_dataset.weights).cuda(),
            ),
            SpatialFormat.TWO_D_CORONAL: DiceCELoss(
                include_background=False,
                softmax=True,
                squared_pred=False,
                weight=torch.tensor(train_dataset.weights).cuda(),
            ),
        }
    elif experiment_type in (
        ExperimentType.TWO_D_CORONAL,
        ExperimentType.TWO_D_AXIAL,
        ExperimentType.TWO_D_SAGITTAL,
    ):
        spatial_format = SpatialFormat(experiment_type.name)
        formats = [spatial_format]
        channel_stack = int(model_config["channel_stack"])

        # Sagittal model used lateral merged labels, others use
        # usual set of labels
        if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
            out_channels = {
                SpatialFormat.TWO_D_SAGITTAL: train_dataset.n_total_merged_labels,
            }
            labels = {
                SpatialFormat.TWO_D_SAGITTAL: train_dataset.merged_labels,
            }
            weights = train_dataset.merged_weights
        else:
            out_channels = {spatial_format: train_dataset.n_total_labels}
            labels = {spatial_format: train_dataset.labels}
            weights = train_dataset.weights

        models = {
            spatial_format: UNet(
                spatial_dims=2,
                in_channels=channel_stack,
                out_channels=out_channels[spatial_format],
                **model_config["unet_params"],
            )
        }
        loss_fn = {
            spatial_format: DiceCELoss(
                include_background=False,
                softmax=True,
                squared_pred=False,
                weight=torch.tensor(weights).cuda(),
            ),
        }
    elif experiment_type == ExperimentType.THREE_D:
        channel_stack = 1
        out_channels = {SpatialFormat.THREE_D: train_dataset.n_total_labels}
        labels = {SpatialFormat.THREE_D: train_dataset.labels}
        models = {
            SpatialFormat.THREE_D: UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=train_dataset.n_total_labels,
                **model_config["unet_params"],
            )
        }
        loss_fn = {
            SpatialFormat.THREE_D: DiceCELoss(
                include_background=False,
                softmax=True,
                squared_pred=False,
                weight=torch.tensor(train_dataset.weights).cuda(),
            )
        }
    else:
        raise ValueError("No such experiment type.")

    for model in models.values():
        model.cuda()

    # Set up optimizer and scheduler
    optimizer_cls = getattr(torch.optim, model_config["optimizer"])
    scheduler_cls = getattr(torch.optim.lr_scheduler, model_config["scheduler"])
    optimizers = {}
    schedulers = {}

    # Set up metrics
    per_class_dice_metrics = {}
    dice_metric = {}
    aggregated_dice_metric = DiceMetric(
        include_background=False,
        num_classes=train_dataset.n_total_labels,
    )
    per_class_aggregated_dice_metrics = {
        c: DiceMetric(include_background=False, num_classes=1)
        for c in train_dataset.labels
    }
    for spatial_format, model in models.items():
        optimizers[spatial_format] = optimizer_cls(
            model.parameters(),
            **model_config["optimizer_params"],
        )
        schedulers[spatial_format] = scheduler_cls(
            optimizers[spatial_format],
            **model_config["scheduler_params"],
        )
        dice_metric[spatial_format] = DiceMetric(
            include_background=False,
            num_classes=len(labels[spatial_format]),
        )
        per_class_dice_metrics[spatial_format] = {
            c: DiceMetric(include_background=False, num_classes=1)
            for c in labels[spatial_format]
        }

    # Set up tensorboard log
    tensorboard_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir)
    weights_dir = output_dir / "checkpoints"
    weights_dir.mkdir(exist_ok=True)

    # Begin loop
    print("Starting training")
    for e in range(model_config["epochs"]):
        for model in models.values():
            model.train()
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        batch_losses: dict[SpatialFormat, list[float]] = {
            spatial_format: [] for spatial_format in models.keys()
        }

        for batch in train_loader:
            for spatial_format in models.keys():
                model = models[spatial_format]
                optimizer = optimizers[spatial_format]

                input_image = reformat_image(
                    batch[train_dataset.image_key],
                    spatial_format,
                    channel_stack=channel_stack,
                )
                if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
                    mask_key = train_dataset.merged_mask_key
                else:
                    mask_key = train_dataset.mask_key
                mask_image = reformat_image(
                    batch[mask_key],
                    spatial_format,
                    channel_stack=1,
                )

                prediction = model(input_image)
                loss = loss_fn[spatial_format](prediction, mask_image)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                batch_losses[spatial_format].append(loss.detach().cpu().item())

            if data_config.get("write_training_images", False):
                gt_labelmaps = batch[train_dataset.mask_key].argmax(
                    axis=1, keepdim=True
                )
                write_batch_multiplanar_tensorboard(
                    writer=writer,
                    tag_group_prefix="train",
                    tag_prefix="train_image",
                    images=batch[train_dataset.image_key],
                    labelmaps=gt_labelmaps,
                    subject_ids=batch["subject_id"],
                    num_classes=train_dataset.n_foreground_labels,
                    step=e,
                )

        for spatial_format in models.keys():
            epoch_loss = np.mean(batch_losses[spatial_format])
            writer.add_scalar(f"loss/train_{spatial_format.name}", epoch_loss, e)
            print(f"Epoch {e}, train loss ({spatial_format.name}) {epoch_loss:.4f}")

        for model in models.values():
            model.eval()
        batch_losses = {spatial_format: [] for spatial_format in models.keys()}
        with torch.no_grad():
            for batch in val_loader:
                gt_labelmaps = batch[val_dataset.mask_key].argmax(axis=1, keepdim=True)

                if e == 0:
                    write_batch_multiplanar_tensorboard(
                        writer=writer,
                        tag_group_prefix="val",
                        tag_prefix="ground_truth",
                        images=batch[val_dataset.image_key],
                        labelmaps=gt_labelmaps,
                        subject_ids=batch["subject_id"],
                        num_classes=val_dataset.n_foreground_labels,
                        step=e,
                    )

                predictions_per_format = {}

                for spatial_format in models.keys():
                    model = models[spatial_format]

                    input_image = reformat_image(
                        batch[val_dataset.image_key],
                        spatial_format,
                        channel_stack=channel_stack,
                    )
                    if spatial_format == SpatialFormat.TWO_D_SAGITTAL:
                        mask_key = val_dataset.merged_mask_key
                    else:
                        mask_key = val_dataset.mask_key
                    mask_image = reformat_image(
                        batch[mask_key],
                        spatial_format,
                        channel_stack=1,
                    )

                    prediction = model(input_image)

                    loss = loss_fn[spatial_format](prediction, mask_image)

                    batch_losses[spatial_format].append(loss.detach().cpu().item())

                    # Move prediction back to 3D format
                    prediction = inverse_reformat_image(
                        prediction.detach(),
                        spatial_format,
                        batch_size,
                    )
                    predictions_per_format[spatial_format] = prediction

                    pred_labelmaps = prediction.argmax(dim=1, keepdim=True)

                    # Calculate metrics in 3D
                    dice_metric[spatial_format](pred_labelmaps, gt_labelmaps)
                    for c, label in enumerate(labels[spatial_format], 1):
                        per_class_dice_metrics[spatial_format][label](
                            pred_labelmaps == c,
                            gt_labelmaps == c,
                        )

                    write_batch_multiplanar_tensorboard(
                        writer=writer,
                        tag_group_prefix="val",
                        tag_prefix=f"prediction_{spatial_format.name}",
                        images=batch[val_dataset.image_key],
                        labelmaps=pred_labelmaps,
                        subject_ids=batch["subject_id"],
                        num_classes=out_channels[spatial_format],
                        step=e,
                    )

                if experiment_type == ExperimentType.TWO_D_ENSEMBLE:
                    # Metrics on the aggregated prediction
                    aggregated_prediction = aggregate_views(
                        axial_pred=predictions_per_format[SpatialFormat.TWO_D_AXIAL],
                        coronal_pred=predictions_per_format[
                            SpatialFormat.TWO_D_CORONAL
                        ],
                        sagittal_pred=predictions_per_format[
                            SpatialFormat.TWO_D_SAGITTAL
                        ],
                        sagittal_unmerging_indices=val_dataset.get_unmerging_indices(),
                    )
                    aggregated_labelmaps = aggregated_prediction.argmax(
                        dim=1, keepdim=True
                    )
                    aggregated_dice_metric(aggregated_labelmaps, gt_labelmaps)
                    for c, label in enumerate(val_dataset.labels, 1):
                        per_class_aggregated_dice_metrics[label](
                            aggregated_labelmaps == c,
                            gt_labelmaps == c,
                        )

                    write_batch_multiplanar_tensorboard(
                        writer=writer,
                        tag_group_prefix="val",
                        tag_prefix="prediction_aggregated",
                        images=batch[val_dataset.image_key],
                        labelmaps=aggregated_labelmaps,
                        subject_ids=batch["subject_id"],
                        num_classes=val_dataset.n_total_labels,
                        step=e,
                    )

        for spatial_format in models.keys():
            epoch_loss = np.mean(batch_losses[spatial_format])
            writer.add_scalar(f"loss/val_{spatial_format.name}", epoch_loss, e)
            writer.add_scalar(
                f"dice_metric_3d/val_{spatial_format.name}",
                dice_metric[spatial_format].aggregate(),
                e,
            )
            dice_metric[spatial_format].reset()

            per_class_averages = []
            for c, label in enumerate(labels[spatial_format], 1):
                val = (
                    per_class_dice_metrics[spatial_format][label]
                    .aggregate()
                    .cpu()  # type: ignore
                    .item()
                )
                writer.add_scalar(
                    f"per_class_dice_metric_3d_{spatial_format.name}/{label}",
                    val,
                    e,
                )
                per_class_dice_metrics[spatial_format][label].reset()
                per_class_averages.append(val)

            writer.add_scalar(
                f"average_dice_metric_3d_{spatial_format.name}",
                np.mean(per_class_averages),
                e,
            )

            print(f"Epoch {e}, val loss ({spatial_format.name}) {epoch_loss:.4f}")

            writer.add_scalar(
                f"learning_rate/{spatial_format.name}",
                schedulers[spatial_format].optimizer.param_groups[0]["lr"],
                e,
            )
            if isinstance(
                schedulers[spatial_format], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                # special case: need to pass loss to step method
                schedulers[spatial_format].step(epoch_loss)
            else:
                schedulers[spatial_format].step()

        if experiment_type == ExperimentType.TWO_D_ENSEMBLE:
            writer.add_scalar(
                "dice_metric_3d/val_aggregated",
                aggregated_dice_metric.aggregate(),
                e,
            )
            aggregated_dice_metric.reset()

            per_class_averages = []
            for c, label in enumerate(val_dataset.labels, 1):
                val = (
                    per_class_aggregated_dice_metrics[label]
                    .aggregate()
                    .cpu()  # type: ignore
                    .item()
                )
                writer.add_scalar(
                    f"per_class_dice_metric_3d_aggregated/{label}",
                    val,
                    e,
                )
                per_class_aggregated_dice_metrics[label].reset()
                per_class_averages.append(val)

            writer.add_scalar(
                "average_dice_metric_3d_aggregated",
                np.mean(per_class_averages),
                e,
            )

        # Save all models together
        model_weights = {
            spatial_format.name: model.state_dict()
            for spatial_format, model in models.items()
        }
        torch.save(model_weights, weights_dir / f"weights_{e:04d}.pt")
