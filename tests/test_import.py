"""Simple tests that the project code can be imported."""
import importlib


def test_import_project():
    """A simple test that the module imported correctly and tests run."""
    importlib.import_module("deepsurfer_train")
