"""Training loop logic."""
from pathlib import Path
from typing import Any

from monai.networks.nets import UNet
from monai.losses import DiceLoss
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


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
    output_dir = Path(output_dir)

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
    model = model.cuda()

    loss_fn = DiceLoss(
        include_background=False,
        softmax=True,
        squared_pred=True,
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

            prediction = model(batch["orig"])
            loss = loss_fn(prediction)

            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(loss.detach().cpu().item())

        epoch_loss = np.mean(batch_losses)
        writer.add_scalar("loss/train", epoch_loss, e)
        print(f"Epoch {e}, train loss {epoch_loss:.4f}")

        model.eval()
        batch_losses = []
        with torch.no_grad():

            for batch in val_loader:

                prediction = model(batch["orig"])
                loss = loss_fn(prediction)

                batch_losses.append(loss.detach().cpu().item())

            epoch_loss = np.mean(batch_losses)
            writer.add_scalar("loss/val", epoch_loss, e)
            print(f"Epoch {e}, val loss {epoch_loss:.4f}")

        writer.add_scalar("learning_rate", scheduler.optimizer.param_groups[0]["lr"], e)
        scheduler.step(epoch_loss)

        torch.save(model.state_dict(), weights_dir / f"weights_{e:04d}.pt")
