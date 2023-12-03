"""Utilities for defining and listed datasets and their files."""
from pathlib import Path

from deepsurfer_train import locations
from deepsurfer_train.enums import BrainRegions, DatasetPartition

DATASETS_AVAILABLE = [
    "buckner40",
]

# List of subjects in each partition
SPLITS = {
    "buckner40": {
        DatasetPartition.TRAIN: [
            "017",
            "021",
            "032",
            "039",
            "040",
            "045",
            "073",
            "080",
            "091",
            "092",
            "093",
            "095",
            "099",
            "102",
            "108",
            "123",
            "130",
            "131",
            "133",
            "141",
        ],
        DatasetPartition.VALIDATION: [
            "004",
            "049",
            "097",
            "103",
            "106",
            "114",
            "128",
            "129",
            "136",
            "138",
        ],
        DatasetPartition.TEST: [
            "008",
            "067",
            "074",
            "084",
            "111",
            "124",
            "140",
            "144",
            "145",
            "149",
        ],
    }
}


# These were calculated over train partition of buckner40
MEDIAN_VOLUMES = {
    BrainRegions.LEFT_CEREBRAL_WHITE_MATTER: 211329.0,
    BrainRegions.LEFT_CEREBRAL_CORTEX: 221814.5,
    BrainRegions.LEFT_LATERAL_VENTRICLE: 12535.5,
    BrainRegions.LEFT_INF_LAT_VENT: 397.0,
    BrainRegions.LEFT_CEREBELLUM_WHITE_MATTER: 12560.5,
    BrainRegions.LEFT_CEREBELLUM_CORTEX: 53347.0,
    BrainRegions.LEFT_THALAMUS_PROPER: 6929.0,
    BrainRegions.LEFT_CAUDATE: 3664.5,
    BrainRegions.LEFT_PUTAMEN: 4828.5,
    BrainRegions.LEFT_PALLIDUM: 1797.0,
    BrainRegions.THIRD_VENTRICLE: 1251.0,
    BrainRegions.FOURTH_VENTRICLE: 1702.5,
    BrainRegions.BRAIN_STEM: 19741.0,
    BrainRegions.LEFT_HIPPOCAMPUS: 3851.5,
    BrainRegions.LEFT_AMYGDALA: 1445.0,
    BrainRegions.CSF: 1085.0,
    BrainRegions.LEFT_VENTRAL_DC: 3794.0,
    BrainRegions.LEFT_ACCUMBENS_AREA: 555.0,
    BrainRegions.LEFT_VESSEL: 27.0,
    BrainRegions.LEFT_CHOROID_PLEXUS: 687.0,
    BrainRegions.RIGHT_CEREBRAL_WHITE_MATTER: 213315.0,
    BrainRegions.RIGHT_CEREBRAL_CORTEX: 222881.5,
    BrainRegions.RIGHT_LATERAL_VENTRICLE: 12768.0,
    BrainRegions.RIGHT_INF_LAT_VENT: 378.5,
    BrainRegions.RIGHT_CEREBELLUM_WHITE_MATTER: 12291.0,
    BrainRegions.RIGHT_CEREBELLUM_CORTEX: 52292.0,
    BrainRegions.RIGHT_THALAMUS_PROPER: 6782.5,
    BrainRegions.RIGHT_CAUDATE: 3836.5,
    BrainRegions.RIGHT_PUTAMEN: 4808.0,
    BrainRegions.RIGHT_PALLIDUM: 1936.5,
    BrainRegions.RIGHT_HIPPOCAMPUS: 3745.5,
    BrainRegions.RIGHT_AMYGDALA: 1586.5,
    BrainRegions.RIGHT_ACCUMBENS_AREA: 548.5,
    BrainRegions.RIGHT_VENTRAL_DC: 3774.5,
    BrainRegions.RIGHT_VESSEL: 18.0,
    BrainRegions.RIGHT_CHOROID_PLEXUS: 751.5,
    BrainRegions.FIFTH_VENTRICLE: 0.0,
    BrainRegions.WM_HYPOINTENSITIES: 1981.0,
    BrainRegions.NON_WM_HYPOINTENSITIES: 0.0,
    BrainRegions.OPTIC_CHIASM: 176.5,
    BrainRegions.CC_POSTERIOR: 997.0,
    BrainRegions.CC_MID_POSTERIOR: 478.0,
    BrainRegions.CC_CENTRAL: 448.0,
    BrainRegions.CC_MID_ANTERIOR: 438.0,
    BrainRegions.CC_ANTERIOR: 940.0,
}
BACKGROUND_MEDIAN_VOLUME = 15669150.0


def get_latest_fs_version(processed_dir: Path | str) -> str:
    """Get the latest freesurfer version used to process a dataset.

    Parameters
    ----------
    processed_dir: Path | str
        Directory containing multiple processed versions of
        a dataset.

    Returns
    -------
    str:
        Name of sub-directory of input directory corresponding
        to the latest freesurfer version that has been used
        to process the dataset.

    """
    processed_dir = Path(processed_dir)
    fs_versions = [v.name for v in processed_dir.glob("fs_v*")]
    return sorted(fs_versions)[-1]


def list_dataset_files(
    dataset: str,
    filenames: dict[str, str],
    partition: DatasetPartition | str | None,
    processed_version: str | None = None,
    root_dir: Path | str = locations.project_dataset_dir,
) -> list[dict[str, Path | str]]:
    """List segmentation files in format needed for monai dataset.

    Parameters
    ----------
    dataset: str
        Name of dataset, e.g. "buckner40"
    filenames: dict[str, str]
        Files required in the dataset, given as dictionaries mapping arbitrary
        keys to paths relative to each subject directory. The chosen keys will
        be the keys used in the returned dictionary. E.g. for the brainmask and
        aseg files, specify the list
        ``{"image": "mri/brainmask.mgz", "mask": "mri/aseg.mgz"}``.
    partition: deepsurfer_train.enums.DatasetPartition | str
        Partition of the dataset (or none, which will return
        entire dataset).
    processed_version: str | None
        Version of processing used to provide images and labels.
        If not specified, the latest version of freesurfer for
        which processed images exist will be used.
    root_dir: Path | str
        The root directory of all datasets.

    Returns
    -------
    list[dict[str, Path]]:
        Dataset definition as required by monai.data.Dataset.
        This consists of a list of dictionaries, one for each
        element of the dataset. Within each dictionary, a key
        maps to a path representing the location of a file.
        Keys of this dict match those in the input ``filenames`` dict.
        Additionally a "subject_id" key gives the subject ID.

    """
    if dataset not in DATASETS_AVAILABLE:
        raise ValueError(f"Dataset {dataset} not recognized!")

    root_dir = Path(root_dir)
    processed_dir = root_dir / dataset / "processed"

    if processed_version is None:
        processed_version = get_latest_fs_version(processed_dir)

    data_dir = processed_dir / processed_version
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if partition is None:
        subjects: list[str] = sum(SPLITS[dataset].values(), [])
    else:
        partition = DatasetPartition(partition)
        subjects = SPLITS[dataset][partition]

    data = []

    for subj in subjects:
        subj_dir = data_dir / subj
        subj_data: dict[str, Path | str] = {}
        for k, p in filenames.items():
            path = subj_dir / p
            if not path.exists():
                raise FileNotFoundError(f"No file found at {str(path)}.")
            subj_data[k] = path
        subj_data["subject_id"] = subj_dir.name
        data.append(subj_data)

    return data
