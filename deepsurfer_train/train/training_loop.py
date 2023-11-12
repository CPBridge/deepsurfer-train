"""Training loop logic."""
from pathlib import Path
from typing import Any

from monai.visualize.utils import blend_images
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


from deepsurfer_train.enums import DatasetPartition
from deepsurfer_train import locations
from deepsurfer_train.preprocess.dataset import DeepsurferSegmentationDataset


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

    # Choose a slice to display
    if isinstance(data_config["imsize"], int):
        display_slice = data_config["imsize"] // 2
    else:
        display_slice = data_config["imsize"][2] // 2

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
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=train_dataset.n_total_labels,
        **model_config["unet_params"],
    )
    model = model.cuda()

    loss_fn = DiceLoss(
        include_background=False,
        softmax=True,
        squared_pred=True,
    )
    dice_metric = DiceMetric(
        include_background=False,
        num_classes=val_dataset.n_total_labels,
    )
    per_class_dice_metrics = {
        val_dataset.get_region_label(c): DiceMetric(
            include_background=False, num_classes=1
        )
        for c in range(1, val_dataset.n_total_labels)
    }

    # Set up optimizer and scheduler
    optimizer_cls = getattr(torch.optim, model_config["optimizer"])
    optimizer = optimizer_cls(
        model.parameters(),
        **model_config["optimizer_params"],
    )
    scheduler_cls = getattr(torch.optim.lr_scheduler, model_config["scheduler"])
    scheduler = scheduler_cls(
        optimizer,
        **model_config["scheduler_params"],
    )

    # Set up tensorboard log
    tensorboard_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir)
    weights_dir = output_dir / "checkpoints"
    weights_dir.mkdir(exist_ok=True)

    # Begin loop
    print("Starting training")
    for e in range(model_config["epochs"]):
        model.train()
        optimizer.zero_grad()
        batch_losses = []

        for batch in train_loader:
            prediction = model(batch[train_dataset.image_key])
            loss = loss_fn(prediction, batch[train_dataset.mask_key])
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(loss.detach().cpu().item())
            if data_config.get("write_training_images", False):
                gt_labelmap = (
                    batch[train_dataset.mask_key].argmax(axis=1, keepdim=True).cpu()
                )
                for b in range(batch_size):
                    scaled_gt_labelmap = (
                        gt_labelmap[b, :, :, :, display_slice]
                        / train_dataset.n_foreground_labels
                    )
                    blend = blend_images(
                        batch[train_dataset.image_key][b, :, :, :, display_slice].cpu(),
                        scaled_gt_labelmap,
                        rescale_arrays=False,
                        cmap="tab20",
                    )
                    writer.add_image(
                        f"train_subject_{batch['subject_id'][b]}/train_image",
                        blend,
                        e,
                    )

        epoch_loss = np.mean(batch_losses)
        writer.add_scalar("loss/train", epoch_loss, e)
        print(f"Epoch {e}, train loss {epoch_loss:.4f}")

        model.eval()
        batch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                prediction = model(batch[val_dataset.image_key])
                loss = loss_fn(prediction, batch[val_dataset.mask_key])

                batch_losses.append(loss.detach().cpu().item())

                pred_labelmap = prediction.argmax(axis=1, keepdim=True).cpu()

                gt_labelmap = (
                    batch[val_dataset.mask_key].argmax(axis=1, keepdim=True).cpu()
                )

                # Calculate metrics
                dice_metric(pred_labelmap, gt_labelmap)
                for c in range(1, val_dataset.n_total_labels):
                    label = val_dataset.get_region_label(c)
                    per_class_dice_metrics[label](
                        pred_labelmap == c,
                        gt_labelmap == c,
                    )

                for b in range(batch_size):
                    scaled_pred_labelmap = (
                        pred_labelmap[b, :, :, :, display_slice]
                        / val_dataset.n_foreground_labels
                    )
                    blend = blend_images(
                        batch[val_dataset.image_key][b, :, :, :, display_slice].cpu(),
                        scaled_pred_labelmap,
                        rescale_arrays=False,
                        cmap="tab20",
                    )
                    writer.add_image(
                        f"val_subject_{batch['subject_id'][b]}/prediction",
                        blend,
                        e,
                    )
                    if e == 0:
                        scaled_gt_labelmap = (
                            gt_labelmap[b, :, :, :, display_slice]
                            / val_dataset.n_foreground_labels
                        )
                        blend = blend_images(
                            batch[val_dataset.image_key][
                                b, :, :, :, display_slice
                            ].cpu(),
                            scaled_gt_labelmap,
                            cmap="tab20",
                            rescale_arrays=False,
                        )
                        writer.add_image(
                            f"val_subject_{batch['subject_id'][b]}/ground_truth",
                            blend,
                            e,
                        )

            epoch_loss = np.mean(batch_losses)
            writer.add_scalar("loss/val", epoch_loss, e)
            writer.add_scalar("dice_metric/val", dice_metric.aggregate(), e)
            dice_metric.reset()
            for c in range(1, val_dataset.n_total_labels):
                label = val_dataset.get_region_label(c)
                writer.add_scalar(
                    f"per_class_dice_metric/{label.name}",
                    per_class_dice_metrics[label].aggregate(),
                    e,
                )
                per_class_dice_metrics[label].reset()

            print(f"Epoch {e}, val loss {epoch_loss:.4f}")

        writer.add_scalar(
            "learning_rate",
            scheduler.optimizer.param_groups[0]["lr"],
            e,
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # special case: need to pass loss to step method
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

        torch.save(model.state_dict(), weights_dir / f"weights_{e:04d}.pt")
