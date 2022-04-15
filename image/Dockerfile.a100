
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04
    
# Install tools and dependencies.
RUN apt-get -y update --fix-missing
RUN apt-get install -y \
  emacs \
  git \
  wget \
  libgoogle-glog-dev

# Setup to install the latest version of cmake.
RUN apt-get install -y software-properties-common && \
    apt-get update && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && apt-get install -y cmake

# Set the working directory.
WORKDIR /root
# install cusparseLT
RUN wget https://developer.download.nvidia.com/compute/libcusparse-lt/0.2.0/local_installers/libcusparse_lt-linux-x86_64-0.2.0.1.tar.gz && \
    tar xzvf libcusparse_lt-linux-x86_64-0.2.0.1.tar.gz

# install sputnik
RUN git clone --recurse-submodules https://github.com/zheng-ningxin/sputnik.git && \
    cd sputnik && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="80" && \
    make -j && cp sputnik/libspmm.so /usr/local/lib/ && cp -r /root/sputnik/third_party/abseil-cpp/absl /usr/local/include/

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
    bash Anaconda3-2021.11-Linux-x86_64.sh -b -p /root/anaconda && \
    eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda create -n artifact python=3.8 -y && \
    conda activate artifact && pip install torch==1.7.0 torchvision==0.8.0 matplotlib pandas

RUN echo 'export CUDA_HOME=/usr/local/cuda \n\
export LD_LIBRARY_PATH=/root/libcusparse_lt/lib64:$LD_LIBRARY_PATH \n\
source ~/anaconda/etc/profile.d/conda.sh \n\
' >> /root/.bashrc
