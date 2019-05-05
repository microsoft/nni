/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

/**
 * GPU Infromation class
 * Representing the dynamic and static information retrieved from Nvidia-smi
 */
export class GPUInfo {
    // The number of active process running on this GPU
    public activeProcessNum: number;
    // Memory utilization of this GPU
    public gpuMemUtil: number;
    // GPU utilization of this GPU
    public gpuUtil: number;
    // the index number of this GPU (starting from 0)
    public readonly index: number;

    constructor(activeProcessNum : number, gpuMemUtil : number, gpuUtil : number, index : number) {
        this.activeProcessNum = activeProcessNum;
        this.gpuMemUtil = gpuMemUtil;
        this.gpuUtil = gpuUtil;
        this.index = index;
    }
}

/**
 * GPU Sumamry for each machine
 */
export class GPUSummary {
    // GPU count on the machine
    public readonly gpuCount: number;
    // The timestamp when GPU summary data queried
    public readonly timestamp: string;
    // The array of GPU information for each GPU card
    public readonly gpuInfos: GPUInfo[];

    constructor(gpuCount: number, timestamp: string, gpuInfos: GPUInfo[]) {
        this.gpuCount = gpuCount;
        this.timestamp = timestamp;
        this.gpuInfos = gpuInfos;
    }
}

export const GPU_INFO_COLLECTOR_FORMAT_LINUX: string = 
`
#!/bin/bash
export METRIC_OUTPUT_DIR={0}
echo $$ >{1}
python3 -m nni_gpu_tool.gpu_metrics_collector
`

export const GPU_INFO_COLLECTOR_FORMAT_WINDOWS: string = 
`
$env:METRIC_OUTPUT_DIR="{0}"
$app = Start-Process "python" -ArgumentList "-m nni_gpu_tool.gpu_metrics_collector" -passthru -NoNewWindow
Write $app.ID | Out-File {1} -NoNewline -encoding utf8
`