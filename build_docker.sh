#!/usr/bin/env bash
# This script is used to build the project docker image, and push it to gitlab
set -e

echo "Building image"
sudo docker build \
    -t deepsurfer-train \
    --build-arg BUILD_COMMIT_HASH=`git rev-parse HEAD` \
    .
