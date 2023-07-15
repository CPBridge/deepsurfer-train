"""Process to set up share directories."""
from pathlib import Path
from typing import Mapping, Sequence

import click

from deepsurfer_train import locations


@click.command()
def setup_project_dir() -> None:
    """Setup directories in project directory.

    Creates all directories contained within the project's locations.py that is
    a directory on the project directory. Any module-scoped variable that is a
    pathlib.Path, or an item of a list of dictionary that is pathlib.Path, will
    be created. If any directory exists, it will not be affected.

    """
    if not locations.project_dir.exists():
        raise RuntimeError(
            "The project data share was not found at "
            f"'{str(locations.project_dir)}.'"
        )

    def _maybe_make_dir(p: Path) -> None:
        # Assume anything with an extension is intended to be a file
        if "." not in p.name:
            if locations.project_dir.resolve() in p.resolve().parents:
                if p.exists():
                    print(f"{p}: already exists, not created")
                else:
                    print(f"{p}: directory created")

    for var, val in locations.__dict__.items():
        if isinstance(val, Path):
            _maybe_make_dir(val)
        elif isinstance(val, Sequence):
            for item in val:
                if isinstance(item, Path):
                    _maybe_make_dir(item)
        elif isinstance(val, Mapping):
            if isinstance(item, Path):
                _maybe_make_dir(item)
