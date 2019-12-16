// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

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

    constructor(activeProcessNum: number, gpuMemUtil: number, gpuUtil: number, index: number) {
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

export const GPU_INFO_COLLECTOR_FORMAT_WINDOWS: string =
`
$env:METRIC_OUTPUT_DIR="{0}"
$app = Start-Process "python" -ArgumentList "-m nni_gpu_tool.gpu_metrics_collector" -passthru -NoNewWindow
Write $app.ID | Out-File {1} -NoNewline -encoding utf8
`;
