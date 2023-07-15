"""Defines locations where project data are stored."""
import sys
from pathlib import Path
from warnings import warn

from importlib_resources import files

# Standard location for share mounting
# This will vary based on the platform
if sys.platform == "darwin":
    # 'darwin' just means MacOS
    # Linux servers and containers
    project_dir = Path("/Volumes/qtim/projects/deepsurfer")
    project_dataset_dir = Path("/Volumes/freesurfer/test_data")
else:
    # Linux servers and containers
    project_dir = Path("/autofs/cluster/qtim/projects/deepsurfer")
    project_dataset_dir = Path("/autofs/cluster/freesurfer/test_data")


# Issue a warning here if the project locations don't exist
if not project_dir.exists():
    warn(
        "The project directory is not found at the expected location: "
        f"{project_dir}.",
        UserWarning,
    )
if not project_dataset_dir.exists():
    warn(
        "The project dataset directory is not found at the expected location: "
        f"{project_dataset_dir}.",
        UserWarning,
    )

# Location to store checkpoints
checkpoints_dir = project_dir / "checkpoints"

# Location to store inferences on train/test/val data
inferences_dir = project_dir / "inferences"

# Location to store results of analyses
analysis_dir = project_dir / "analysis"

# Location to store visualizations
vis_dir = project_dir / "vis"

# Location where any resources within the package codebase are installed
resources_dir = files("deepsurfer_train") / "resources"
# Location where train configs are kept
train_configs_dir = resources_dir / "training_configs"
