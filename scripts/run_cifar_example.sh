#!/bin/bash
Docker/docker_build.sh
docker run -it --rm -p 8888:8888 \
  -v $(pwd):/code \
  --entrypoint /code/scripts/cifar_example.sh \
  iclr_aug
