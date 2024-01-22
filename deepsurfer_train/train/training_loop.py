"""Training loop logic."""
from pathlib import Path
from typing import Any

from monai.metrics.meandice import DiceMetric
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter


from deepsurfer_train.enums import (
    DatasetPartition,
)
from deepsurfer_train import locations
from deepsurfer_train.preprocess.dataset import DeepsurferSegmentationDataset
from deepsurfer_train.visualization.tensorboard import (
    write_batch_multiplanar_tensorboard,
)
from deepsurfer_train.methods.three_d_unet import UNet3DMethod
from deepsurfer_train.methods.two_d_unet_ensemble import UNet2DEnsembleMethod
from deepsurfer_train.methods.bayesnet import BayesNetMethod


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

    # Set up metrics
    per_class_dice_metrics = {}
    dice_metric = DiceMetric(
        include_background=False,
        num_classes=train_dataset.n_total_labels,
    )
    per_class_dice_metrics = {
        c: DiceMetric(include_background=False, num_classes=1)
        for c in train_dataset.labels
    }

    # Set up tensorboard log
    tensorboard_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir)
    weights_dir = output_dir / "checkpoints"
    weights_dir.mkdir(exist_ok=True)

    method_name = model_config["method"]
    method_cls = {
        "UNet2DEnsembleMethod": UNet2DEnsembleMethod,
        "UNet3DMethod": UNet3DMethod,
        "BayesNetMethod": BayesNetMethod,
    }[method_name]
    method = method_cls(  # type: ignore
        model_config=model_config,
        method_params=model_config["method_params"],
        writer=writer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Begin loop
    print("Starting training")
    for e in range(model_config["epochs"]):

        method.train_begin()
        batch_losses = []

        for batch in train_loader:

            loss = method.train_step(batch)
            batch_losses.append(loss)

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

        epoch_loss = np.mean(batch_losses)
        writer.add_scalar("loss/train", epoch_loss, e)
        print(f"Epoch {e}, train loss {epoch_loss:.4f}")

        method.train_end(e)
        method.val_begin(e)

        batch_losses = []
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

                val_loss, pred_labelmaps = method.val_step(batch, e)
                batch_losses.append(val_loss)

                # Calculate metrics in 3D
                dice_metric(pred_labelmaps, gt_labelmaps)
                for c, label in enumerate(val_dataset.labels, 1):
                    per_class_dice_metrics[label](
                        pred_labelmaps == c,
                        gt_labelmaps == c,
                    )

                write_batch_multiplanar_tensorboard(
                    writer=writer,
                    tag_group_prefix="val",
                    tag_prefix="prediction",
                    images=batch[val_dataset.image_key],
                    labelmaps=pred_labelmaps,
                    subject_ids=batch["subject_id"],
                    num_classes=val_dataset.n_total_labels,
                    step=e,
                )

        epoch_loss = np.mean(batch_losses)
        writer.add_scalar("loss/val", epoch_loss, e)
        writer.add_scalar(
            "dice_metric_3d/val",
            dice_metric.aggregate(),
            e,
        )
        dice_metric.reset()

        per_class_averages = []
        for label in val_dataset.labels:
            val = per_class_dice_metrics[label].aggregate().cpu().item()  # type: ignore
            writer.add_scalar(
                f"per_class_dice_metric_3d/{label}",
                val,
                e,
            )
            per_class_dice_metrics[label].reset()
            per_class_averages.append(val)

        writer.add_scalar(
            "average_dice_metric_3d",
            np.mean(per_class_averages),
            e,
        )

        print(f"Epoch {e}, val loss {epoch_loss:.4f}")

        method.val_end(e, float(epoch_loss))

        # Save all models together
        model_weights = method.get_state_dict()
        torch.save(model_weights, weights_dir / f"weights_{e:04d}.pt")
