"""Implementation details of the FastSurfer models."""
import torch


def aggregate_views(
    axial_pred: torch.Tensor,
    coronal_pred: torch.Tensor,
    sagittal_pred: torch.Tensor,
    sagittal_unmerging_indices: list[int],
) -> torch.Tensor:
    """Aggregrate 2D predictions from the three planes.

    This follows the "fastsurfer" implementation.

    Henschel et al 2020. "FastSurfer - A fast and accurate deep learning based
    neuroimaging pipeline"

    Parameters
    ----------
    axial_pred: torch.Tensor
        Prediction from the axial model. This is a tensor in PLI 3D format (B x
        C x H x W x D) reformatted from the output of an axial 2D model. The
        tensor contains the raw logit outputs before softmax or thresholding.
    coronal_pred: torch.Tensor
        Prediction from the axial model. This is a tensor in PLI 3D format (B x
        C x H x W x D) reformatted from the output of a coronal 2D model. The
        tensor contains the raw logit outputs before softmax or thresholding.
    sagittal_pred: torch.Tensor
        Prediction from the sagittal model. This is a tensor in PLI 3D format
        (B x C x H x W x D) reformatted from the output of a sagittal 2D model.
        The tensor contains the raw logit outputs before softmax or
        thresholding.
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
    b, c, h, w, d = axial_pred.shape
    if coronal_pred.shape != (b, c, h, w, d):
        raise ValueError("Shape of axial and coronal tensors must match.")
    if sagittal_pred.shape[0] != b:
        raise ValueError("Batch size of sagittal and axial tensors must match.")
    if sagittal_pred.shape[2:] != (h, w, d):
        raise ValueError("Spatial shape of sagittal and axial tensors must match.")
    if len(sagittal_unmerging_indices) != c:
        raise ValueError(
            "Length of sagittal unmerging indices must be equal to channels in "
            "axial and coronal images."
        )

    unmerged_sagittal_pred = sagittal_pred[:, sagittal_unmerging_indices, :, :, :]

    aggregated = (
        torch.softmax(axial_pred, dim=1)
        + torch.softmax(coronal_pred, dim=1)
        + 0.5 * torch.softmax(unmerged_sagittal_pred, dim=1)
    ) / 2.5

    return aggregated
