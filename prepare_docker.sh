#!/bin/bash

docker run -it --rm --gpus all --name Hunyuan3d \
  --privileged \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /storage_local/kwang/repos/Hunyuan3D-2:/workspace \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -w /workspace \
  --entrypoint /bin/bash \
  hunyuan3d:latest -c "
    # Create mount point
    mkdir -p /data/nuscenes
    # Mount the squashfs to temporary location
    squashfuse /data/nuscenes.sqfs /data/nuscenes
    # Start interactive bash session
    exec /bin/bash
  "

