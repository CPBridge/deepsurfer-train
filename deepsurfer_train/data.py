"""Utilities for defining and listed datasets and their files."""
from pathlib import Path

from deepsurfer_train import locations
from deepsurfer_train.enums import DatasetPartition

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


def list_segmentation_files(
    dataset: str,
    partition: DatasetPartition | str | None,
    processed_version: str | None = None,
    root_dir: Path | str = locations.project_dataset_dir,
) -> list[dict[str, Path | str]]:
    """List segmentation files in format needed for monai dataset.

    Parameters
    ----------
    dataset: str
        Name of dataset, e.g. "bucker40"
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
        Included keys are "orig" and "aseg". Additionally a
        "subject_id" key gives the subject ID.

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
        orig = subj_dir / "mri" / "orig.mgz"
        aseg = subj_dir / "mri" / "aseg.mgz"
        subj_data: dict[str, Path | str] = {"orig": orig, "aseg": aseg}
        for path in [orig, aseg]:
            if not path.exists():
                raise FileNotFoundError(f"No file found at {str(path)}.")
        subj_data["subject_id"] = subj_dir.name
        data.append(subj_data)

    return data
