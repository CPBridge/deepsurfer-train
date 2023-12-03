"""Dataset class."""
from pathlib import Path
from typing import Sequence

import monai
import pandas as pd
import numpy as np
import torch
from typeguard import typechecked

from deepsurfer_train import locations
from deepsurfer_train.data import (
    list_dataset_files,
    MEDIAN_VOLUMES,
    BACKGROUND_MEDIAN_VOLUME,
)
from deepsurfer_train.enums import (
    DatasetPartition,
    BrainRegions,
)
from deepsurfer_train.preprocess.transforms import (
    RandomizableIdentityd,
    SynthTransformd,
    VoxynthAugmentd,
)


DEFAULT_EXCLUDED_REGIONS = (
    "FIFTH_VENTRICLE",
    "NON_WM_HYPOINTENSITIES",
    "LEFT_VESSEL",
    "RIGHT_VESSEL",
)


def get_label_mapping(
    excluded_regions: Sequence[BrainRegions] | None = None,
) -> pd.DataFrame:
    """Get a mapping between original label and "internal" label.

    The "internal" set of labels is consecutive, starting at 1, and omits
    any excluded labels. The "merged internal" set of labels is similar
    but additionally merges lateralized labels into a single label.

    Parameters
    ----------
    excluded_regions: Sequence[deepsurfer_train.enums.BrainRegions] | None
        Regions to exclude from the output labels.

    Returns
    -------
    pandas.DataFrame
        Mapping of all original label values to internal labels and
        merged internal labels. Contains the following columns:
        "name", "original_value, "internal_value", "merged_name",
        "merged_internal_value".

    """

    def merge_name(name: str) -> str:
        for lat_str in ["LEFT_", "RIGHT_"]:
            if name.startswith(lat_str):
                return name.split(lat_str, maxsplit=1)[1]
        return name

    # Rows will be appended here
    mapping: list[dict[str, int | str]] = []
    next_label = 1
    next_merged_label = 1

    # Dictionary of merged label names to values
    merged_labels: dict[str, int] = {}

    background_row = {
        "name": "BACKGROUND",
        "original_value": 0,
        "internal_value": 0,
        "original_volume": BACKGROUND_MEDIAN_VOLUME,
        "merged_name": "BACKGROUND",
        "merged_internal_value": 0,
    }
    mapping.append(background_row)

    excluded_regions = [] if excluded_regions is None else excluded_regions
    for label in BrainRegions:
        row = {"name": label.name, "original_value": label.value}
        row["original_volume"] = MEDIAN_VOLUMES[label]
        if label in excluded_regions:
            # Map this value to zero
            row["internal_value"] = 0
            row["merged_internal_value"] = 0
        else:
            row["internal_value"] = next_label
            merged_name = merge_name(label.name)
            next_label += 1

            row["merged_name"] = merged_name
            if merged_name in merged_labels:
                row["merged_internal_value"] = merged_labels[merged_name]
            else:
                row["merged_internal_value"] = next_merged_label
                merged_labels[merged_name] = next_merged_label
                next_merged_label += 1

        mapping.append(row)

    mapping_df = pd.DataFrame(mapping)

    # Calculate weights for each internal class as inversely proportional
    # to the median volume, rescaled to sum to 1
    mapping_df["weight"] = pd.Series(0.0, index=mapping_df.index)
    weight_sum = 0.0
    for v in range(mapping_df.internal_value.max() + 1):
        source_rows = mapping_df[mapping_df.internal_value == v]
        internal_volume = source_rows.original_volume.sum()
        weight = 1.0 / internal_volume
        mapping_df.loc[
            mapping_df.internal_value == v,
            "weight"
        ] = weight
        weight_sum += weight
    mapping_df["weight"] = mapping_df["weight"] / weight_sum

    # Repeat for the merged classes
    mapping_df["merged_weight"] = pd.Series(0.0, index=mapping_df.index)
    weight_sum = 0.0
    for v in range(mapping_df.merged_internal_value.max() + 1):
        source_rows = mapping_df[mapping_df.merged_internal_value == v]
        merged_internal_volume = source_rows.original_volume.sum()
        weight = 1.0 / merged_internal_volume
        mapping_df.loc[
            mapping_df.merged_internal_value == v,
            "merged_weight"
        ] = weight
        weight_sum += weight
    mapping_df["merged_weight"] = mapping_df["merged_weight"] / weight_sum

    return mapping_df


@typechecked
class DeepsurferSegmentationDataset(monai.data.CacheDataset):
    """Dataset for use with segmentation problems."""

    def __init__(
        self,
        dataset: str,
        partition: DatasetPartition | str | None,
        processed_version: str | None = None,
        root_dir: Path | str = locations.project_dataset_dir,
        image_file: str = "mri/brainmask.mgz",
        mask_file: str = "mri/aseg.mgz",
        imsize: Sequence[int] | int = 255,
        excluded_regions: Sequence[BrainRegions | str]
        | None = DEFAULT_EXCLUDED_REGIONS,
        use_spatial_augmentation: bool = False,
        use_intensity_augmentation: bool = False,
        synth_probability: float = 0.0,
        use_gpu: bool = True,
    ):
        """Dataset class encapsulate loading and transforming data.

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
        excluded_regions: Sequence[deepsurfer_train.enums.BrainRegions | str] | None
            Brain regions to omit from the segmentation masks.
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
        image_key = "image"
        mask_key = "mask"
        merged_mask_key = "merged_mask"
        elements_list = list_dataset_files(
            dataset=dataset,
            filenames={
                image_key: image_file,
                mask_key: mask_file,
            },
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

        excluded_regions = [] if excluded_regions is None else excluded_regions
        excluded_regions_ = [
            r if isinstance(r, BrainRegions) else BrainRegions[r]
            for r in excluded_regions
        ]
        self.label_mapping = get_label_mapping(excluded_regions_)

        load_keys = [image_key, mask_key]
        all_keys = [image_key, mask_key, merged_mask_key]

        transforms: list[monai.transforms.MapTransform] = []

        transforms.append(
            monai.transforms.LoadImaged(
                keys=load_keys,
                ensure_channel_first=True,
                simple_keys=True,
                reader="NibabelReader",
                image_only=True,
            )
        )
        transforms.extend(
            [
                monai.transforms.MapLabelValued(
                    keys=mask_key,
                    orig_labels=self.label_mapping.original_value.values.tolist(),
                    target_labels=self.label_mapping.internal_value.values.tolist(),
                ),
                monai.transforms.CopyItemsd(
                    keys=[mask_key],
                    names=[merged_mask_key],
                ),
                monai.transforms.MapLabelValued(
                    keys=merged_mask_key,
                    orig_labels=self.label_mapping.internal_value.values.tolist(),
                    target_labels=self.label_mapping.merged_internal_value.values.tolist(),
                ),
            ]
        )

        # Scale intensity from input range
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

        if use_gpu:
            device = torch.device("cuda:0")
            # This prevents items being cached on the GPU
            transforms.append(
                RandomizableIdentityd(keys=[image_key, mask_key, "subject_id"])
            )
            transforms.append(monai.transforms.ToDeviced(device=device, keys=all_keys))
        else:
            device = torch.device("cpu")

        transforms.append(
            monai.transforms.Orientationd(
                keys=all_keys,
                axcodes="PLI",
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
                    mode=["bilinear", "nearest", "nearest"],
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
                        mask_key=merged_mask_key,
                        image_output_keys=[image_key],
                        apply_probability=synth_probability,
                    )
                )

            transforms.append(
                VoxynthAugmentd(
                    keys=[image_key],
                    mask_key=None,
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

        transforms.extend(
            [
                monai.transforms.AsDiscreted(
                    keys=[mask_key],
                    to_onehot=int(self.label_mapping.internal_value.max()) + 1,
                ),
                monai.transforms.AsDiscreted(
                    keys=[merged_mask_key],
                    to_onehot=int(self.label_mapping.merged_internal_value.max()) + 1,
                ),
            ]
        )

        composed_transforms = monai.transforms.Compose(transforms, lazy=True)

        super().__init__(elements_list, composed_transforms)

        # A transform that can be used to map the data back to the original
        non_omitted_mapping = self.label_mapping[self.label_mapping.internal_value > 0]
        self.inverse_label_map_transform = monai.transforms.MapLabelValue(
            orig_labels=non_omitted_mapping.internal_value.values.tolist(),
            target_labels=non_omitted_mapping.original_value.values.tolist(),
        )
        self.image_key = image_key
        self.mask_key = mask_key
        self.merged_mask_key = merged_mask_key

    def get_region_label_enum(self, label: int) -> BrainRegions:
        """Get the original label for an internal label pixel value.

        Parameters
        ----------
        label: int
            Internal label value (used in segmentation masks).

        Returns
        -------
        deepsurfer_train.enums.BrainRegions:
            Original label corresponding to the input label.

        """
        row = self.label_mapping[self.label_mapping.internal_value == label].iloc[0]
        return BrainRegions(row.original_value)

    def get_region_label(self, label: int) -> str:
        """Get the original label for an internal label pixel value.

        Parameters
        ----------
        label: int
            Internal label value (used in segmentation masks).

        Returns
        -------
        str:
            Original label name corresponding to the input label.

        """
        row = self.label_mapping[self.label_mapping.internal_value == label].iloc[0]
        return row.name

    def get_merged_region_label(self, label: int) -> str:
        """Get the original label for a merged internal label pixel value.

        Parameters
        ----------
        label: int
            Internal label value (used in segmentation masks).

        Returns
        -------
        str:
             Label name corresponding to the merged input label.

        """
        row = self.label_mapping[
            self.label_mapping.merged_internal_value == label
        ].iloc[0]
        return row.merged_name

    @property
    def labels(self) -> list[str]:
        """List of all labels used, in order, excluding background (0)."""
        return self.label_mapping[
            self.label_mapping.internal_value > 0
        ].sort_values("internal_value").name.values.tolist()

    @property
    def merged_labels(self) -> list[str]:
        """List of all merged labels used, in order, excluding background (0)."""
        return (
            self.label_mapping[self.label_mapping.merged_internal_value > 0]
            .sort_values("merged_internal_value")
            .name.values.tolist()
        )

    @property
    def n_foreground_labels(self) -> int:
        """int: Number of foreground labels in segmentation masks."""
        return int(self.label_mapping.internal_value.max())

    @property
    def n_merged_foreground_labels(self) -> int:
        """int: Number of merged foreground labels in segmentation masks."""
        return int(self.label_mapping.merged_internal_value.max())

    @property
    def n_total_labels(self) -> int:
        """int: Number of total labels (inc background) in segmentation masks."""
        return self.n_foreground_labels + 1

    @property
    def n_total_merged_labels(self) -> int:
        """int: Number of total labels (inc background) in segmentation masks."""
        return self.n_merged_foreground_labels + 1

    @property
    def weights(self) -> list[float]:
        """Weights for each internal label (including background)."""
        return [
            self.label_mapping[self.label_mapping.internal_value == v].iloc[0].weight
            for v in range(self.label_mapping.internal_value.max() + 1)
        ]

    @property
    def merged_weights(self) -> list[float]:
        """Weights for each merged internal label (including background)."""
        return [
            self.label_mapping[self.label_mapping.merged_internal_value == v].iloc[0].merged_weight
            for v in range(self.label_mapping.merged_internal_value.max() + 1)
        ]

    def get_unmerging_indices(self) -> list[int]:
        """Get indices to use to undo the label merging.

        Returns
        -------
        list[int]:
            List of integers that, when applied to the channel dimension of a
            one-hot encoded array of the merged label set (with lateral
            structures merged into a single label), creates a one-hot encoded
            array of the unmerged internal labels. This is used to map the
            sagittal model's outputs to match the shape of the other models
            in preparation for merging.

        """
        df = self.label_mapping[self.label_mapping.internal_value > 0].sort_values(
            "internal_value"
        )
        return [0] + df.merged_internal_value.values.tolist()
