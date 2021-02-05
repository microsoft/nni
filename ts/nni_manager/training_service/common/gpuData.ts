// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

export enum ScheduleResultType {
    // Schedule succeeded
    SUCCEED,

    // Temporarily, no enough available GPU right now
    TMP_NO_AVAILABLE_GPU,

    // Cannot match requirement even if all GPU are a
    REQUIRE_EXCEED_TOTAL
}

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
    public gpuMemTotal: number;
    public gpuMemFree: number;
    public gpuMemUsed: number;
    public gpuType: string;

    constructor(activeProcessNum: number, gpuMemUtil: number, gpuUtil: number, index: number,
        gpuMemTotal: number, gpuMemFree: number, gpuMemUsed: number, gpuType: string) {
        this.activeProcessNum = activeProcessNum;
        this.gpuMemUtil = gpuMemUtil;
        this.gpuUtil = gpuUtil;
        this.index = index;
        this.gpuMemTotal = gpuMemTotal;
        this.gpuMemFree = gpuMemFree;
        this.gpuMemUsed = gpuMemUsed;
        this.gpuType = gpuType;
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


export function parseGpuIndices(gpuIndices?: string): Set<number> | undefined {
    if (gpuIndices !== undefined) {
        const indices: number[] = gpuIndices.split(',')
            .map((x: string) => parseInt(x, 10));
        if (indices.length > 0) {
            return new Set(indices);
        } else {
            throw new Error('gpuIndices can not be empty if specified.');
        }
    }
}

export const GPU_INFO_COLLECTOR_FORMAT_WINDOWS: string =
    `
$env:METRIC_OUTPUT_DIR="{0}"
$app = Start-Process "python" -ArgumentList "-m nni.tools.gpu_tool.gpu_metrics_collector" -passthru -NoNewWindow
Write $app.ID | Out-File {1} -NoNewline -encoding utf8
`;
