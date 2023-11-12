"""Utilities for writing visualizations to tensorboard."""
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from monai.visualize.utils import blend_images


def write_tensorboard_overlay(
    writer: SummaryWriter,
    tag: str,
    image: torch.Tensor,
    labelmap: torch.Tensor,
    num_classes: int,
    step: int,
) -> None:
    """Write a 2D mask/image overlay to tensorboard.

    Parameters
    ----------
    writer: torch.utils.tensorboard.SummaryWriter
        Writer to write the image to.
    tag: str
        Tag to record in tensorboard.
    image: torch.Tensor
        Image tensor in format (C H W) with a single channel.
    labelmap: torch.Tensor
        Mask tensor in format (C H W) with a single channel
        and "labelmap" values.
    num_classes: int
        Number of foreground classes in the mask.
    step: int
        Global step number.

    """
    scaled_labelmap = labelmap / num_classes
    blend = blend_images(
        image.cpu(),
        scaled_labelmap.cpu(),
        rescale_arrays=False,
        cmap="tab20",
    )
    writer.add_image(tag, blend, step)


def write_multiplanar_tensorboard(
    writer: SummaryWriter,
    tag_group: str,
    tag_prefix: str,
    image: torch.Tensor,
    labelmap: torch.Tensor,
    num_classes: int,
    step: int,
) -> None:
    """Write center-slice overlays in three planes to tensorboard.

    Parameters
    ----------
    writer: torch.utils.tensorboard.SummaryWriter
        Writer to write the image to.
    tag_group: str
        Tag group to record in tensorboard.
    tag: str
        Tag to record in tensorboard.
    image: torch.Tensor
        Image tensor in format (C H W D) with a single channel. Assumes
        PLI orientation.
    labelmap: torch.Tensor
        Mask tensor in format (C H W D) with a single channel
        and "labelmap" values. Assumes PLI orientation.
    num_classes: int
        Number of foreground classes in the mask.
    step: int
        Global step number.

    """
    # coronal
    s = image.shape[1] // 2
    write_tensorboard_overlay(
        writer=writer,
        tag=f"{tag_group}/{tag_prefix}_coronal",
        image=image[:, s, :, :].permute([0, 2, 1]),
        labelmap=labelmap[:, s, :, :].permute([0, 2, 1]),
        num_classes=num_classes,
        step=step,
    )

    # sagittal
    s = image.shape[2] // 2
    write_tensorboard_overlay(
        writer=writer,
        tag=f"{tag_group}/{tag_prefix}_sagittal",
        image=image[:, :, s, :].permute([0, 2, 1]),
        labelmap=labelmap[:, :, s, :].permute([0, 2, 1]),
        num_classes=num_classes,
        step=step,
    )

    # axial
    s = image.shape[3] // 2
    write_tensorboard_overlay(
        writer=writer,
        tag=f"{tag_group}/{tag_prefix}_axial",
        image=image[:, :, :, s],
        labelmap=labelmap[:, :, :, s],
        num_classes=num_classes,
        step=step,
    )


def write_batch_multiplanar_tensorboard(
    writer: SummaryWriter,
    tag_group_prefix: str,
    tag_prefix: str,
    images: torch.Tensor,
    labelmaps: torch.Tensor,
    subject_ids: list[str],
    num_classes: int,
    step: int,
) -> None:
    """Write tensorboard visualizations for an entire batch.

    Parameters
    ----------
    writer: torch.utils.tensorboard.SummaryWriter
        Writer to write the image to.
    model_name: str
        Name of the model to record in the tag name.
    tag_group_prefix: str
        Prefix of tag group for tensorboard, will be combined with
        subject id.
    tag_prefix: str
        Prefix of tag for tensorboard, will be combined with
        orientation.
    image: torch.Tensor
        Image tensor in format (B C H W D) with a single channel.
    labelmap: torch.Tensor
        Mask tensor in format (B C H W D) with a single channel
        and "labelmap" values.
    subject_ids: list[str]
        List of patient identifiers.
    num_classes: int
        Number of foreground classes in the mask.
    step: int
        Global step number.

    """
    for b in range(images.shape[0]):
        write_multiplanar_tensorboard(
            writer=writer,
            tag_group=f"{tag_group_prefix}_{subject_ids[b]}",
            tag_prefix=tag_prefix,
            image=images[b],
            labelmap=labelmaps[b],
            num_classes=num_classes,
            step=step,
        )
