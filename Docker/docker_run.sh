#!/bin/bash
docker run -it --rm -p 8888:8888 \
  -v $(pwd):/code \
  mkuchnik/iclr19_aug \
  "/bin/bash"
