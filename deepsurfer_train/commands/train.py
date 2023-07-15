"""Main training functions."""
import json
from pathlib import Path
from typing import Optional

import click
from pycrumbs import tracked

from deepsurfer_train import locations


@click.command()
@click.argument("config_file")
@click.argument("model_name")
@click.option("--seed", "-s", type=int, help="Random seed to use.")
@tracked(
    literal_directory=locations.checkpoints_dir,
    subdirectory_name_parameter="model_name",
    seed_parameter="seed",
    include_uuid=True,
    directory_injection_parameter="model_output_dir",
)
def train(
    config_file: str,
    model_name: str,
    model_output_dir: Path,
    seed: Optional[int] = None,
) -> None:
    """Train a model.

    This will create a new directory in the checkpoints directory containing
    model weights and any other files associated with model training.

    CONFIG_FILE is the name of the config file within the repo's training
    config directory containing all parameters for the training job.

    MODEL_NAME is a free text string that names the model. The output will be
    placed in a directory with this name.

    """
    # Read in the config file
    if not config_file.lower().endswith(".json"):
        config_file += ".json"
    config_path = locations.train_configs_dir / config_file
    with config_path.open("r") as jf:
        config = json.load(jf)

    # Stash the configuration in the output directory
    config_copy = model_output_dir / "train_config.json"
    with config_copy.open("w") as jf:
        json.dump(config, jf, indent=4)

    # Actual training code goes here...
    pass
