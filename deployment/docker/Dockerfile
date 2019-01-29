# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

LABEL maintainer='Microsoft NNI Team<nni@microsoft.com>'

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y install sudo \
    apt-utils \
    git \
    curl \
    vim \
    unzip \
    wget \
    build-essential \ 
    cmake \
    libopenblas-dev \
    automake \
    openssh-client \
    openssh-server \
    lsof \
    python3.5 \
    python3-dev \
    python3-pip \
    python3-tk \
    libcupti-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#
# update pip
#
RUN python3 -m pip install --upgrade pip

# numpy 1.14.3  scipy 1.1.0 
RUN python3 -m pip --no-cache-dir install \
    numpy==1.14.3 scipy==1.1.0 

#
# Tensorflow 1.10.0
#
RUN python3 -m pip --no-cache-dir install tensorflow-gpu==1.10.0

#
# Keras 2.1.6
#
RUN python3 -m pip --no-cache-dir install Keras==2.1.6

#
# PyTorch 
#
RUN python3 -m pip --no-cache-dir install torch==0.4.1
RUN python3 -m pip install torchvision==0.2.1

#
# sklearn 0.20.0
#
RUN python3 -m pip --no-cache-dir install scikit-learn==0.20.0

#
# pandas==0.23.4 lightgbm==2.2.2
#
RUN python3 -m pip --no-cache-dir install pandas==0.23.4 lightgbm==2.2.2

#
# Install NNI
#
RUN python3 -m pip --no-cache-dir install nni

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbin

WORKDIR /root
