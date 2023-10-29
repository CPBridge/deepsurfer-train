"""Training loop logic."""
from pathlib import Path
from typing import Any

from monai.networks.nets import UNet
import torch


from deepsurfer_train.enums import DatasetPartition
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

    # Set up dataset
    train_dataset = DeepsurferSegmentationDataset(
        dataset=data_config["dataset"],
        partition=DatasetPartition.TRAIN,
        processed_version=data_config.get("processed_version", None),
        root_dir=data_config.get("root_dir", None),
        imsize=data_config["imsize"],
        use_spatial_augmentation=True,
        use_intensity_augmentation=True,
        synth_probability=data_config["synth_probability"],
        use_gpu=use_gpu,
    )
    val_dataset = DeepsurferSegmentationDataset(
        dataset=data_config["dataset"],
        partition=DatasetPartition.VALIDATION,
        processed_version=data_config.get("processed_version", None),
        root_dir=data_config.get("root_dir", None),
        imsize=data_config["imsize"],
        use_spatial_augmentation=False,
        use_intensity_augmentation=False,
        synth_probability=0.0,  # ?
        use_gpu=use_gpu,
    )

    # Set up data loader
    batch_size = model_config["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Set up model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=train_dataset.n_total_labels,
        **model_config["unet_params"],
    )

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

    # Begin loop
