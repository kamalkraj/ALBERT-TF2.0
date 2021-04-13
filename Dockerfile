FROM tensorflow/tensorflow:2.4.1-gpu

# set working directory
WORKDIR /albert

# update pip
RUN apt-get update && apt-get install -y python3-pip nano && /usr/bin/python3 -m pip install -U pip

# install requirements
RUN pip install sentencepiece==0.1.83 tqdm==4.60.0

# add source code
COPY . /albert