#!/usr/bin/env bash
# This script is used to run the docker image as interactive container for
# development
set -e

SCRIPT_LOCATION=$(readlink -f $BASH_SOURCE)
HOST_REPO=$(dirname $SCRIPT_LOCATION)
CONTAINER_REPO="/deepsurfer-train"
echo "The repository will appear inside the container as ${CONTAINER_REPO}"
echo "The project share will appear inside the container as ${CONTAINER_SHARE_MOUNT}"

sudo docker run \
    -it \
    --rm \
    --name deepsurfer-train-dev \
    -u $(id -u):$(id -g) \
    -v ${HOST_REPO}:${CONTAINER_REPO} \
    -v /autofs/:/autofs/ \
    -w ${CONTAINER_REPO} \
    deepsurfer-train \
    /bin/bash
