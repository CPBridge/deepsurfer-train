"""Dataset class."""
from pathlib import Path
from typing import Sequence

import monai
import numpy as np
import torch
from typeguard import typechecked

from deepsurfer_train import locations
from deepsurfer_train.data import list_segmentation_files
from deepsurfer_train.enums import DatasetPartition
from deepsurfer_train.preprocess.transforms import (
    SynthTransformd,
    VoxynthAugmentd,
)


@typechecked
class DeepsurferSegmentationDataset(monai.data.Dataset):
    """Dataset for use with segmentation problems."""

    def __init__(
        self,
        dataset: str,
        partition: DatasetPartition | str | None,
        processed_version: str | None = None,
        root_dir: Path | str = locations.project_dataset_dir,
        imsize: Sequence[int] | int = 255,
        use_spatial_augmentation: bool = False,
        use_intensity_augmentation: bool = False,
        synth_probability: float = 0.0,
        use_gpu: bool = True,
    ):
        """Dataset class encapsulate loading and tranforming data.

        Parameters
        ----------
        dataset: str
            Name of dataset, e.g. "buckner40"
        partition: deepsurfer_train.enums.DatasetPartition | str
            Partition of the dataset (or none, which will return
            entire dataset).
        processed_version: str | None
            Version of processing used to provide images and labels.
            If not specified, the latest version of freesurfer for
            which processed images exist will be used.
        root_dir: Path | str
            The root directory of all datasets.
        imsize: Sequence[int] | int
            Image size in pixels.
        use_spatial_augmentation: bool
            Whether to augment images and masks with spatial transforms.
        use_intensity_augmentation: bool
            Whether to augment images and masks with intensity transforms.
        synth_probability: float
            Probability of applying the synth transform to the mask.
        use_gpu: bool
            Use the GPU for all preprocessing.

        """
        partition = DatasetPartition(partition)
        elements_list = list_segmentation_files(
            dataset=dataset,
            partition=partition,
            processed_version=processed_version,
            root_dir=root_dir,
        )
        if isinstance(imsize, int):
            imsize = [imsize] * 3
        if len(imsize) != 3:
            raise ValueError("Imsize must have length 3.")

        if synth_probability < 0.0 or synth_probability > 1.0:
            raise ValueError(
                "Argument 'synth_probability' must be between 0.0 and 1.0."
            )

        image_key = "orig"
        mask_key = "aseg"
        all_keys = [image_key, mask_key]

        transforms: list[monai.transforms.MapTransform] = []

        transforms.append(
            monai.transforms.LoadImaged(
                keys=all_keys,
                ensure_channel_first=True,
                simple_keys=True,
                reader="NibabelReader",
                image_only=True,
            )
        )

        if use_gpu:
            device = torch.device("cuda:0")
            transforms.append(monai.transforms.ToDeviced(device=device, keys=all_keys))
        else:
            device = torch.device("cpu")

        # Scale intensity from input range (assumed to be 0 to
        # _max_input_intensity), to 0-1
        transforms.append(
            monai.transforms.ScaleIntensityRanged(
                keys=[image_key],
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )

        transforms.append(
            monai.transforms.Orientationd(
                keys=all_keys,
                axcodes="PIL",
                lazy=True,
            )
        )

        transforms.append(
            monai.transforms.ResizeWithPadOrCropd(
                keys=all_keys,
                spatial_size=imsize,
                lazy=True,
            )
        )

        if use_spatial_augmentation:
            transforms.append(
                monai.transforms.RandAffined(
                    keys=all_keys,
                    prob=0.75,
                    rotate_range=np.pi / 8.0,
                    shear_range=np.pi / 16.0,
                    translate_range=10.0,
                    scale_range=0.2,
                    mode=["bilinear", "nearest"],
                    padding_mode="zeros",
                    lazy=True,
                    spatial_size=imsize,
                    cache_grid=True,
                    device=device,
                )
            )

        if use_intensity_augmentation:
            if synth_probability > 0.0:
                transforms.append(
                    SynthTransformd(
                        mask_key=mask_key,
                        image_output_keys=[image_key],
                        apply_probability=synth_probability,
                    )
                )

            transforms.append(
                VoxynthAugmentd(
                    keys=[image_key],
                    mask_key=mask_key,
                    bias_field_probability=0.2,
                    inversion_probability=0.0,
                    smoothing_probability=0.2,
                    smoothing_one_axis_probability=0.2,
                    background_noise_probability=0.0,
                    background_blob_probability=0.0,
                    added_noise_probability=0.5,
                    added_noise_max_sigma=0.1,
                    wave_artifact_probability=0.2,
                    line_corruption_probability=0.2,
                    gamma_scaling_probability=0.2,
                    resized_one_axis_probability=0.2,
                )
            )

        composed_transforms = monai.transforms.Compose(transforms)

        super().__init__(elements_list, composed_transforms)
