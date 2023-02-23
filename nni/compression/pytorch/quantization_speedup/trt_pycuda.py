#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# NOTE: this file copies partial content of https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common.py

import pycuda.driver as cuda
import pycuda.autoinit        # pylint: disable=unused-import
import tensorrt as trt

def explicit_batch():
    return 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        Simple helper data class that's a little nicer to use than a 2-tuple.

        Parameters
        ----------
        host_mem : host memory
            Memory buffers of host
        device_mem : device memory
            Memory buffers of device
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """
    Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    NOTE: currently this function only supports NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag.

    Parameters
    ----------
    engine : tensorrt.ICudaEngine
        An ICudaEngine for executing inference on a built network

    Returns
    -------
    list
        All input HostDeviceMem of an engine
    list
        All output HostDeviceMem of an engine
    GPU bindings
        Device bindings
    GPU stream
        A stream is a sequence of commands (possibly issued by different host threads) that execute in order
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) # * engine.max_batch_size, batch size already in
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]