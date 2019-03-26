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

import { TrialJobDetail, TrialJobStatus } from '../../common/trainingService';
import { GPUScheduler } from './gpuScheduler';
import { LocalTrainingService } from './localTrainingService';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';

type LocalTrialJobDetailForGPU = TrialJobDetail & { gpuIndices: number[] };

/**
 * Local training service for GPU
 */
class LocalTrainingServiceForGPU extends LocalTrainingService {
    private requiredGPUNum!: number;
    private gpuScheduler!: GPUScheduler;
    private availableGPUIndices: boolean[];

    constructor() {
        super();
        this.availableGPUIndices = Array(16).fill(false); // Assume the maximum gpu number is 16
    }

    public async run(): Promise<void> {
        if (this.gpuScheduler !== undefined) {
            await Promise.all([
                this.gpuScheduler.run(),
                super.run()
            ]);
        } else {
            await super.run();
        }
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        await super.setClusterMetadata(key, value);
        switch (key) {
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if(this.localTrailConfig !== undefined) {
                    this.requiredGPUNum = this.localTrailConfig.gpuNum;
                } else {
                    // If no valid trial config is initialized, set requiredGPUNum to 0 as fallback value.
                    this.requiredGPUNum = 0;
                }
                this.log.info('required GPU number is ' + this.requiredGPUNum);
                if (this.gpuScheduler === undefined && this.requiredGPUNum > 0) {
                    this.gpuScheduler = new GPUScheduler();
                }
                break;
            default:
        }
    }

    public cleanUp(): Promise<void> {
        if (this.gpuScheduler !== undefined) {
            this.gpuScheduler.stop();
        }

        return super.cleanUp();
    }

    protected onTrialJobStatusChanged(trialJob: LocalTrialJobDetailForGPU, oldStatus: TrialJobStatus): void {
        super.onTrialJobStatusChanged(trialJob, oldStatus);
        if (trialJob.gpuIndices !== undefined && trialJob.gpuIndices.length !== 0 && this.gpuScheduler !== undefined) {
            if (oldStatus === 'RUNNING' && trialJob.status !== 'RUNNING') {
                for (const index of trialJob.gpuIndices) {
                    this.availableGPUIndices[index] = false;
                }
            }
        }
    }

    protected getEnvironmentVariables(
        trialJobDetail: TrialJobDetail,
        resource: { gpuIndices: number[] }): { key: string; value: string }[] {
        const variables: { key: string; value: string }[] = super.getEnvironmentVariables(trialJobDetail, resource);
        variables.push({
            key: 'CUDA_VISIBLE_DEVICES',
            value: this.gpuScheduler === undefined ? '' : resource.gpuIndices.join(',')
        });

        return variables;
    }

    protected setExtraProperties(trialJobDetail: LocalTrialJobDetailForGPU, resource: { gpuIndices: number[] }): void {
        super.setExtraProperties(trialJobDetail, resource);
        trialJobDetail.gpuIndices = resource.gpuIndices;
    }

    protected tryGetAvailableResource(): [boolean, {}] {
        const [success, resource] = super.tryGetAvailableResource();
        if (!success || this.gpuScheduler === undefined) {
            return [success, resource];
        }

        const availableGPUIndices: number[] = this.gpuScheduler.getAvailableGPUIndices();
        const selectedGPUIndices: number[] = availableGPUIndices.filter((index: number) => this.availableGPUIndices[index] === false);

        if (selectedGPUIndices.length < this.requiredGPUNum) {
            return [false, resource];
        }

        selectedGPUIndices.splice(this.requiredGPUNum);
        Object.assign(resource, { gpuIndices: selectedGPUIndices });

        return [true, resource];
    }

    protected occupyResource(resource: { gpuIndices: number[] }): void {
        super.occupyResource(resource);
        if (this.gpuScheduler !== undefined) {
            for (const index of resource.gpuIndices) {
                this.availableGPUIndices[index] = true;
            }
        }
    }
}

export { LocalTrainingServiceForGPU };
