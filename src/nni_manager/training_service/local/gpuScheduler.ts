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

import * as cpp from 'child-process-promise';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { String } from 'typescript-string-operations';
import { getLogger, Logger } from '../../common/log';
import { delay } from '../../common/utils';
import { GPU_INFO_COLLECTOR_FORMAT, GPUInfo, GPUSummary } from '../common/gpuData';

/**
 * GPUScheduler for local training service
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

    public getAvailableGPUIndices(): number[] {
        if (this.gpuSummary !== undefined) {
            return this.gpuSummary.gpuInfos.filter((info: GPUInfo) => info.activeProcessNum === 0)
                .map((info: GPUInfo) => info.index);
        }

        return [];
    }

    public async stop(): Promise<void> {
        this.stopping = true;
        try {
            const pid: string = await fs.promises.readFile(path.join(this.gpuMetricCollectorScriptFolder, 'pid'), 'utf8');
            await cpp.exec(`pkill -P ${pid}`);
            await cpp.exec(`rm -rf ${this.gpuMetricCollectorScriptFolder}`);
        } catch (error) {
            this.log.error(`GPU scheduler error: ${error}`);
        }
    }

    /**
     * Generate gpu metric collector shell script in local machine,
     * used to run in remote machine, and will be deleted after uploaded from local.
     */
    private async runGpuMetricsCollectorScript(): Promise<void> {
        await cpp.exec(`mkdir -p ${this.gpuMetricCollectorScriptFolder}`);
        //generate gpu_metrics_collector.sh
        const gpuMetricsCollectorScriptPath: string = path.join(this.gpuMetricCollectorScriptFolder, 'gpu_metrics_collector.sh');
        const gpuMetricsCollectorScriptContent: string = String.Format(
            GPU_INFO_COLLECTOR_FORMAT,
            this.gpuMetricCollectorScriptFolder,
            path.join(this.gpuMetricCollectorScriptFolder, 'pid')
        );
        await fs.promises.writeFile(gpuMetricsCollectorScriptPath, gpuMetricsCollectorScriptContent, { encoding: 'utf8' });
        cp.exec(`bash ${gpuMetricsCollectorScriptPath}`);
    }

    private async updateGPUSummary(): Promise<void> {
        const cmdresult: cpp.childProcessPromise.Result =
            await cpp.exec(`tail -n 1 ${path.join(this.gpuMetricCollectorScriptFolder, 'gpu_metrics')}`);
        if (cmdresult && cmdresult.stdout) {
            this.gpuSummary = <GPUSummary>JSON.parse(cmdresult.stdout);
        } else {
            this.log.error('Could not get gpu metrics information!');
        }
    }
}

export { GPUScheduler };
