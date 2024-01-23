"""Package code for the deepsurfer-train project."""
import os


# Set environment vars correctly for voxelmorpth and neurite
# Do it here so this this occurs before those modules are imported
os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"
