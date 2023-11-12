"""Main training functions."""
import json
from pathlib import Path
from typing import Optional

import click
from pycrumbs import tracked

from deepsurfer_train import locations
from deepsurfer_train.train.training_loop import training_loop


@click.command()
@click.argument("data_config_file")
@click.argument("model_config_file")
@click.argument(
    "model_output_dir",
    type=click.Path(path_type=Path),
)
@click.option("--seed", "-s", type=int, help="Random seed to use.")
@tracked(
    directory_parameter="model_output_dir",
    seed_parameter="seed",
    include_uuid=True,
)
def train(
    data_config_file: Path,
    model_config_file: Path,
    model_output_dir: Path,
    seed: Optional[int] = None,
) -> None:
    """Train a model.

    This will create a new directory in the checkpoints directory containing
    model weights and any other files associated with model training.

    DATA_CONFIG_FILE contains parameters of the dataset, and MODEL_CONFIG_FILE
    contains parameters of the model and training process. Both are specified
    as names (without path) of the config file within the repo's training
    config directory.

    All files resulting from the training are placed in the
    MODEL_OUTPUT_DIRECTORY.

    """
    # Read in the model config file
    if not model_config_file.lower().endswith(".json"):
        model_config_file += ".json"
    model_config_path = locations.train_configs_dir / model_config_file
    with model_config_path.open("r") as jf:
        model_config = json.load(jf)

    # Stash the configuration in the output directory
    model_config_copy = model_output_dir / "model_config.json"
    with model_config_copy.open("w") as jf:
        json.dump(model_config, jf, indent=4)

    # Read in the data config file
    if not data_config_file.lower().endswith(".json"):
        data_config_file += ".json"
    data_config_path = locations.train_configs_dir / data_config_file
    with data_config_path.open("r") as jf:
        data_config = json.load(jf)

    # Stash the configuration in the output directory
    data_config_copy = model_output_dir / "data_config.json"
    with data_config_copy.open("w") as jf:
        json.dump(data_config, jf, indent=4)

    # Actual training code goes here...
    training_loop(
        data_config=data_config,
        model_config=model_config,
        output_dir=model_output_dir,
    )
