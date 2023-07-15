#!/usr/bin/env bash

# Export poetry as requirements.txt
poetry export -o requirements.txt --with dev --without-hashes

# Build the singularity image
OUTPUT_FILE="/autofs/cluster/qtim/projects/deepsurfer/containers/singularity.sif"
singularity build --fakeroot $OUTPUT_FILE singularity.recipe
echo "Built image at $OUTPUT_FILE"

# Cleanup
rm requirements.txt
