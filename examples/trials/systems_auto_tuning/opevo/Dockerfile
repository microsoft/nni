FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV HIP_PLATFORM hcc
ENV PATH $PATH:/opt/rocm/bin:/usr/local/nvidia/lib64/bin
ENV TVM_HOME=/opt/tvm
ENV PYTHONPATH=/usr/local/rocm/src:$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python
ENV HSA_USERPTR_FOR_PAGED_MEM=0

RUN env > /etc/environment

RUN apt-get update && apt install -y --no-install-recommends git ca-certificates \
    python3-pip python3-wheel python3-setuptools python3-dev python3-pytest \
    vim less netcat-openbsd inetutils-ping curl patch iproute2 \
    g++ libpci3 libnuma-dev make cmake file openssh-server kmod gdb libopenmpi-dev openmpi-bin \
        autoconf automake autotools-dev libtool multiarch-support \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    printf "deb [arch=amd64] http://repo.radeon.com/rocm/apt/3.3/ xenial main" | tee /etc/apt/sources.list.d/rocm_hip.list && \
    apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    rocm-dev zlib1g-dev unzip librdmacm-dev rocblas hipsparse rccl rocfft rocrand miopen-hip && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN ln -sf libcudart.so /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart_static.a

RUN pip3 install tornado psutil xgboost==0.80 numpy decorator attrs && rm -rf ~/.cache
RUN git clone https://github.com/dmlc/tvm $TVM_HOME

RUN cd $TVM_HOME && git checkout v0.6 && git submodule init && git submodule update && \
    mkdir -p build && cd build && cp ../cmake/config.cmake . && \
    sed -i 's/LLVM ON/LLVM OFF/g' config.cmake && sed -i 's/CUDA OFF/CUDA ON/g' config.cmake && \
    cmake .. && make -j16

RUN pip3 install nni==1.5 && rm -rf ~/.cache
RUN pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && rm -rf ~/.cache

ADD tvm_patches/tvm_v0.6.patch $TVM_HOME/tvm_v0.6.patch
ADD tvm_patches/libcuda.so.1 $TVM_HOME/build
RUN ln -sf libcuda.so.1 $TVM_HOME/build/libcudart.so.10.0
RUN cd $TVM_HOME && git apply tvm_v0.6.patch && cd build && make -j16

ADD src /root/

