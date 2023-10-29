"""Location to store enumerations associated with your project.

Enums should be used wherever there are a limited number of discrete
values for a certain variable. This is preferable to using simple
strings, since errors are likely to be found earlier, it is clear
what options are available, and enums may be iterated through.

"""
from enum import Enum


class DatasetPartition(Enum):
    """Enum listing partitions of the dataset."""

    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class BrainRegions(Enum):
    """Enum for regions in the aseg.mgz file."""

    # https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/
    # FreeSurferColorLUT
    LEFT_CEREBRAL_WHITE_MATTER = 2
    LEFT_CEREBRAL_CORTEX = 3
    LEFT_LATERAL_VENTRICLE = 4
    LEFT_INF_LAT_VENT = 5
    LEFT_CEREBELLUM_WHITE_MATTER = 7
    LEFT_CEREBELLUM_CORTEX = 8
    LEFT_THALAMUS_PROPER = 10
    LEFT_CAUDATE = 11
    LEFT_PUTAMEN = 12
    LEFT_PALLIDUM = 13
    THIRD_VENTRICLE = 14
    FOURTH_VENTRICLE = 15
    BRAIN_STEM = 16
    LEFT_HIPPOCAMPUS = 17
    LEFT_AMYGDALA = 18
    CSF = 24
    LEFT_VENTRAL_DC = 28
    LEFT_ACCUMBENS_AREA = 26
    LEFT_VESSEL = 30
    LEFT_CHOROID_PLEXUS = 31
    RIGHT_CEREBRAL_WHITE_MATTER = 41
    RIGHT_CEREBRAL_CORTEX = 42
    RIGHT_LATERAL_VENTRICLE = 43
    RIGHT_INF_LAT_VENT = 44
    RIGHT_CEREBELLUM_WHITE_MATTER = 46
    RIGHT_CEREBELLUM_CORTEX = 47
    RIGHT_THALAMUS_PROPER = 49
    RIGHT_CAUDATE = 50
    RIGHT_PUTAMEN = 51
    RIGHT_PALLIDUM = 52
    RIGHT_HIPPOCAMPUS = 53
    RIGHT_AMYGDALA = 54
    RIGHT_ACCUMBENS_AREA = 58
    RIGHT_VENTRAL_D_C = 60
    RIGHT_VESSEL = 62
    RIGHT_CHOROIDP_PEXUS = 63
    FIFTH_VENTRICLE = 72
    WM_HYPOINTENSITIES = 77
    NON_WM_HYPOINTENSITIES = 80
    OPTIC_CHIASM = 85
    CC_POSTERIOR = 251
    CC_MID_POSTERIOR = 252
    CC_CENTRAL = 253
    CC_MID_ANTERIOR = 254
    CC_ANTERIOR = 255
