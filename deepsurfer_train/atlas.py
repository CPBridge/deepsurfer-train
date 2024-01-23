"""Config and methods related to atlases."""
from typing import Sequence
import monai
import numpy as np
import torch
from typeguard import typechecked

from deepsurfer_train.enums import BrainRegions


DEFAULT_ATLAS_PATH = "/space/pac/1/users/dfp15/data/prob_atlas.npz"

# List of classes in the list above
# Note that the background class (0) is not included
DEFAULT_ATLAS_FOREGROUND_CLASSES = [
    BrainRegions.LEFT_CEREBRAL_CORTEX,
    BrainRegions.CSF,
    BrainRegions.LEFT_CEREBRAL_WHITE_MATTER,
    BrainRegions.RIGHT_CEREBRAL_CORTEX,
    BrainRegions.RIGHT_CEREBRAL_WHITE_MATTER,
    BrainRegions.RIGHT_CEREBELLUM_CORTEX,
    BrainRegions.LEFT_CEREBELLUM_CORTEX,
    BrainRegions.RIGHT_CEREBELLUM_WHITE_MATTER,
    BrainRegions.LEFT_CEREBELLUM_WHITE_MATTER,
    BrainRegions.RIGHT_LATERAL_VENTRICLE,
    BrainRegions.RIGHT_CHOROID_PLEXUS,
    BrainRegions.WM_HYPOINTENSITIES,
    BrainRegions.LEFT_LATERAL_VENTRICLE,
    BrainRegions.LEFT_CHOROID_PLEXUS,
    BrainRegions.FOURTH_VENTRICLE,
    BrainRegions.BRAIN_STEM,
    BrainRegions.LEFT_HIPPOCAMPUS,
    BrainRegions.RIGHT_HIPPOCAMPUS,
    BrainRegions.LEFT_INF_LAT_VENT,
    BrainRegions.RIGHT_INF_LAT_VENT,
    BrainRegions.LEFT_THALAMUS_PROPER,
    BrainRegions.RIGHT_THALAMUS_PROPER,
    BrainRegions.LEFT_VENTRAL_DC,
    BrainRegions.RIGHT_VENTRAL_DC,
    BrainRegions.THIRD_VENTRICLE,
    BrainRegions.RIGHT_PUTAMEN,
    BrainRegions.LEFT_PUTAMEN,
    BrainRegions.LEFT_CAUDATE,
    BrainRegions.RIGHT_CAUDATE,
    BrainRegions.LEFT_PALLIDUM,
    BrainRegions.RIGHT_PALLIDUM,
    BrainRegions.RIGHT_VESSEL,
    BrainRegions.NON_WM_HYPOINTENSITIES,
    BrainRegions.LEFT_AMYGDALA,
    BrainRegions.RIGHT_AMYGDALA,
    BrainRegions.LEFT_VESSEL,
    BrainRegions.OPTIC_CHIASM,
    BrainRegions.RIGHT_ACCUMBENS_AREA,
    BrainRegions.LEFT_ACCUMBENS_AREA,
    BrainRegions.FIFTH_VENTRICLE,
]


@typechecked
def get_atlas_metatensor(
    regions: Sequence[BrainRegions] | None = None,
) -> monai.data.MetaTensor:
    """Returns an atlas in monai metatensor format.

    Parameters
    ----------
    regions: Sequence[BrainRegions] | None
        Ordered list of regions to include in the atlas. The regions will
        appear stacked in this order along the channel dimension of the output
        tensor. The background channel is always included implicitly at index 0
        and therefore should not be included in this list.

    Returns
    -------
    monai.data.MetaTensor:
        Loaded atlas as a metatensor, with correctly populated affine and
        channel information.

    """
    atlas_arr = np.load(DEFAULT_ATLAS_PATH)["arr_0"]

    if regions is not None:
        indices = [0]  # background class is always at index 0
        for region in regions:
            try:
                index = DEFAULT_ATLAS_FOREGROUND_CLASSES.index(region)
            except ValueError as e:
                raise ValueError(f"Region {region.name} is not in the atlas.") from e
            indices.append(index)

        # Index down channels of the original array
        atlas_arr = atlas_arr[:, :, :, indices]

    # For whatever reason, this atlas is in LIA convention
    # and has its channels stacked down axis 3
    affine = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    atlas_t = monai.data.MetaTensor(
        atlas_arr,
        affine=affine,
        meta={"original_channel_dim": 3},
    )
    return atlas_t
