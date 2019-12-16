// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { getLogger, Logger } from '../../common/log';
import { delay } from '../../common/utils';
import { GPUInfo, GPUSummary } from '../common/gpuData';
import { execKill, execMkdir, execRemove, execTail, runGpuMetricsCollector } from '../common/util';

/**
 * GPUScheduler for local training service
 */
class GPUScheduler {

    private gpuSummary!: GPUSummary;
    private stopping: boolean;
    private readonly log: Logger;
    private readonly gpuMetricCollectorScriptFolder: string;

    constructor() {
        this.stopping = false;
        this.log = getLogger();
        this.gpuMetricCollectorScriptFolder = `${os.tmpdir()}/${os.userInfo().username}/nni/script`;
    }

    public async run(): Promise<void> {
        await this.runGpuMetricsCollectorScript();
        while (!this.stopping) {
            try {
                await this.updateGPUSummary();
            } catch (error) {
                this.log.error('Read GPU summary failed with error: ', error);
            }
            if (this.gpuSummary !== undefined && this.gpuSummary.gpuCount === 0) {
                throw new Error('GPU not available. Please check your CUDA configuration');
            }
            await delay(5000);
        }
    }

    public getAvailableGPUIndices(useActiveGpu: boolean, occupiedGpuIndexNumMap: Map<number, number>): number[] {
        if (this.gpuSummary !== undefined) {
            if (process.platform === 'win32' || useActiveGpu) {
                return this.gpuSummary.gpuInfos.map((info: GPUInfo) => info.index);
            } else {
                return this.gpuSummary.gpuInfos.filter((info: GPUInfo) =>
                         occupiedGpuIndexNumMap.get(info.index) === undefined && info.activeProcessNum === 0 ||
                         occupiedGpuIndexNumMap.get(info.index) !== undefined)
                       .map((info: GPUInfo) => info.index);
            }
        }

        return [];
    }

    public getSystemGpuCount(): number {
        if (this.gpuSummary !== undefined) {
            return this.gpuSummary.gpuCount;
        }

        return 0;
    }

    public async stop(): Promise<void> {
        this.stopping = true;
        try {
            const pid: string = await fs.promises.readFile(path.join(this.gpuMetricCollectorScriptFolder, 'pid'), 'utf8');
            await execKill(pid);
            await execRemove(this.gpuMetricCollectorScriptFolder);
        } catch (error) {
            this.log.error(`GPU scheduler error: ${error}`);
        }
    }

    /**
     * Generate gpu metric collector shell script in local machine,
     * used to run in remote machine, and will be deleted after uploaded from local.
     */
    private async runGpuMetricsCollectorScript(): Promise<void> {
        await execMkdir(this.gpuMetricCollectorScriptFolder, true);
        runGpuMetricsCollector(this.gpuMetricCollectorScriptFolder);
    }

    private async updateGPUSummary(): Promise<void> {
        const gpuMetricPath: string = path.join(this.gpuMetricCollectorScriptFolder, 'gpu_metrics');
        if (fs.existsSync(gpuMetricPath)) {
            const cmdresult: cpp.childProcessPromise.Result = await execTail(gpuMetricPath);
            if (cmdresult !== undefined && cmdresult.stdout !== undefined) {
                this.gpuSummary = <GPUSummary>JSON.parse(cmdresult.stdout);
            } else {
                this.log.error('Could not get gpu metrics information!');
            }
        } else {
            this.log.warning('gpu_metrics file does not exist!');
        }
    }
}

export { GPUScheduler };
