# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG NNI_RELEASE

LABEL maintainer='Microsoft NNI Team<nni@microsoft.com>'

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get --allow-unauthenticated update
RUN apt-get -y install wget

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation
RUN apt-key del 7fa2af80
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm cuda-keyring_1.0-1_all.deb

RUN apt-get -y update
RUN apt-get -y install \
    automake \
    build-essential \
    cmake \
    curl \
    git \
    openssh-server \
    python3 \
    python3-dev \
    python3-pip \
    sudo \
    unzip \
    zip
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN ln -s python3 /usr/bin/python

RUN python3 -m pip --no-cache-dir install pip==22.0.3 setuptools==60.9.1 wheel==0.37.1

RUN python3 -m pip --no-cache-dir install \
    lightgbm==3.3.2 \
    numpy==1.22.2 \
    pandas==1.4.1 \
    scikit-learn==1.0.2 \
    scipy==1.8.0

RUN python3 -m pip --no-cache-dir install \
    torch==1.10.2+cu113 \
    torchvision==0.11.3+cu113 \
    torchaudio==0.10.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN python3 -m pip --no-cache-dir install pytorch-lightning==1.6.1

RUN python3 -m pip --no-cache-dir install tensorflow==2.9.1

RUN python3 -m pip --no-cache-dir install azureml==0.2.7 azureml-sdk==1.38.0

COPY dist/nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl .
RUN python3 -m pip install nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl
RUN rm nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl

ENV PATH=/root/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/usr/sbin

WORKDIR /root
