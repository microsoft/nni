// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { getLogger } from 'common/log';
import { runPythonModule } from 'common/pythonScript';

const logger = getLogger('GpuInfoCollector');

export interface GpuSystemInfo {
    success: boolean;
    gpuNumber: number;
    gpus: GpuInfo[];
    processes: ProcessInfo[];
    timestamp: number;
}

export interface GpuInfo {
    index: number;
    gpuCoreUtilization: number;
    gpuMemoryUtilization: number;
}

export interface ProcessInfo {
    gpuIndex: number;
    type: 'compute' | 'graphics';
}

let cache: GpuSystemInfo | null = null;
const minUpdateInterval = 10 * 1000;

export async function collectGpuInfo(forceUpdate?: boolean): Promise<GpuSystemInfo | null> {
    if (!forceUpdate && cache !== null) {
        if (Date.now() - cache.timestamp < minUpdateInterval) {
            return cache;
        }
    }

    let str: string;
    try {
        const args = (forceUpdate ? [ '--detail' ] : undefined);
        str = await runPythonModule('nni.tools.nni_manager_scripts.collect_gpu_info', args);
    } catch (error) {
        logger.error('Failed to collect GPU info:', error);
        return null;
    }

    let info: GpuSystemInfo;
    try {
        info = JSON.parse(str);
    } catch (error) {
        logger.error('Failed to collect GPU info, collector output:', str);
        return null;
    }

    if (!info.success) {
        logger.error('Failed to collect GPU info, collector output:', info);
        return null
    }

    if (forceUpdate) {
        logger.info('Forced update:', info);
    } else {
        logger.debug(info);
    }

    cache = info;
    return info;
}
