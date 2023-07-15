#!/usr/bin/env bash¬

SIF_FILE="/autofs/cluster/qtim/projects/deepsurfer/singularity.sif"
singularity shell \
    --nv \
    --writable-tmpfs \
    -B /autofs/:/autofs/ \
    $SIF_FILE
