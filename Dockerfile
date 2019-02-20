FROM tensorflow/tensorflow:1.12.0-py3

LABEL maintainer="Michael Kuchnik <michaelkuchnik@gmail.com>"

RUN mkdir $HOME/code
WORKDIR $HOME/code
COPY requirements.txt $HOME/code/requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y \
  libsm6 \
  libxext6 \
  octave \
  wget \
&& rm -rf /var/lib/apt/lists/*