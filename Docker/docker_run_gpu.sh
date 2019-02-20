#!/bin/bash
nvidia-docker run --runtime=nvidia -it --rm -p 8888:8888 \
  -v $(pwd):/code \
  mkuchnik/iclr19_aug:gpu \
  "/bin/bash"
