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

import { delay } from '../../common/utils';
import { GPUInfo, GPUSummary } from '../common/gpuData';
import { getLogger, Logger } from '../../common/log';
import * as cp from 'child_process';
import * as cpp from 'child-process-promise';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import { String } from 'typescript-string-operations';
import { GPU_COLLECTOR_FORMAT } from '../common/gpuData'

/**
 * GPUScheduler
 */
class GPUScheduler {

    private gpuSummary!: GPUSummary;
    private stopping: boolean;
    private log: Logger;
    private gpuMetricCollectorScriptFolder: string;

    constructor() {
        this.stopping = false;
        this.log = getLogger();
        this.gpuMetricCollectorScriptFolder = `${os.tmpdir()}/nni/script`;
    }

    public async run(): Promise<void> {
        await this.runGpuMetricsCollectorScript();
        while (!this.stopping) {
            try {
                await this.updateGPUSummary();
            } catch (error) {
                this.log.error('Read GPU summary failed with error: ', error);
            }
            await delay(5000);
        }
    }

    /**
     * Generate gpu metric collector shell script in local machine, 
     * used to run in remote machine, and will be deleted after uploaded from local. 
     */
    private async runGpuMetricsCollectorScript(): Promise<void> {
        await cpp.exec(`mkdir -p ${this.gpuMetricCollectorScriptFolder}`);
        //generate gpu_metrics_collector.sh
        let gpuMetricsCollectorScriptPath: string = path.join(this.gpuMetricCollectorScriptFolder, 'gpu_metrics_collector.sh');
        const gpuMetricsCollectorScriptContent: string = String.Format(
            GPU_COLLECTOR_FORMAT,
            this.gpuMetricCollectorScriptFolder,
            path.join(this.gpuMetricCollectorScriptFolder, 'pid'),
        );
        await fs.promises.writeFile(gpuMetricsCollectorScriptPath, gpuMetricsCollectorScriptContent, { encoding: 'utf8' });
        cp.exec(`bash ${gpuMetricsCollectorScriptPath}`);
    }

    public getAvailableGPUIndices(): number[] {
        if (this.gpuSummary !== undefined) {
            return this.gpuSummary.gpuInfos.filter((info: GPUInfo) => info.activeProcessNum === 0).map((info: GPUInfo) => info.index);
        }

        return [];
    }

    public async stop() {
        this.stopping = true;
        const pid: string = await fs.promises.readFile(path.join(this.gpuMetricCollectorScriptFolder, 'pid'), 'utf8');
        console.log(pid)
        await cpp.exec(`pkill -P ${pid}`);
    }

    private async updateGPUSummary() {
        const cmdresult = await cpp.exec(`tail -n 1 ${path.join(this.gpuMetricCollectorScriptFolder, 'gpu_metrics')}`);
        if(cmdresult && cmdresult.stdout) {
            this.gpuSummary = <GPUSummary>JSON.parse(cmdresult.stdout);
        }
    }
}

export { GPUScheduler };
