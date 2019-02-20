#!/bin/bash
Docker/docker_build_gpu.sh
nvidia-docker run --runtime=nvidia -it --rm -p 8888:8888 \
  -v $(pwd):/code \
  --entrypoint /code/scripts/cifar_example.sh \
  iclr_aug:gpu

