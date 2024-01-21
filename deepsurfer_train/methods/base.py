"""Protocol base class for a segmentation method."""
from typing import Any, Protocol

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from deepsurfer_train.preprocess.dataset import DeepsurferSegmentationDataset


class MethodProtocol(Protocol):
    """A protocol describing segmentation methods.

    Each method should implement several callbacks to plug into the main
    training loop.

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
        pass

    def train_begin(self) -> None:
        """Callback called at the beginning of a training epoch.

        This should do things like move models to train mode and reset training
        statistics.

        """
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def get_state_dict(self) -> dict[str, Any]:
        """Get the state dict for all models used by the method.

        Parameters
        ----------
        dict[str, Any]:
            State dict to save.

        """
        pass
