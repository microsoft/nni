// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { getLogger } from 'common/log';
import { GpuSystemInfo, collectGpuInfo } from './collect_info';

const logger = getLogger('GpuScheduler');

export interface ScheduleRestrictions {
    onlyAcceptIndices?: number[];
    rejectActiveGpus?: boolean;
    rejectComputeActiveGpus?: boolean;
}

interface SchedulerGpuInfo {
    index: number;
    util: number;
    coreUtil: number;
    memUtil: number;
    active: boolean;
    computeActive: boolean;
}

interface SchedulerTrialInfo {
    gpuIndex: number;
    experimentId: string;
    trialId: string;
    util: number;
}

export class GpuScheduler {
    private gpus: SchedulerGpuInfo[] = [];
    private trials: SchedulerTrialInfo[] = [];

    public async init(): Promise<void> {
        const info = await collectGpuInfo(true);
        if (info === null) {
            throw new Error('GpuScheduler: Failed to collect GPU info');
        }
        if (info.gpuNumber === 0) {
            throw new Error('GpuScheduler: No GPU found');
        }

        for (let i = 0; i < info.gpuNumber; i++) {
            this.gpus.push({
                index: i,
                util: 0,
                coreUtil: 0,
                memUtil: 0,
                active: false,
                computeActive: false,
            });
        }

        this.updateGpus(info);
    }

    public async update(forceUpdate?: boolean): Promise<void> {
        const info = await collectGpuInfo(forceUpdate);
        if (info === null) {
            if (forceUpdate) {
                throw new Error('GpuScheduler: Failed to update GPU info');
            }
            return;
        }
        if (info.gpuNumber !== this.gpus.length) {
            throw new Error(`GpuScheduler: GPU number changed from ${this.gpus.length} to ${info.gpuNumber}`);
        }

        this.updateGpus(info);
    }

    public async schedule(
            experimentId: string,
            trialId: string,
            gpuNumber: number,
            restrictions: ScheduleRestrictions): Promise<number[] | null> {

        if (gpuNumber >= this.gpus.length) {
            throw new Error(`GpuScheduler: Only have ${this.gpus.length} GPUs, requesting ${gpuNumber}`);
        }

        const gpus = this.sortGpus(restrictions);

        if (gpuNumber < 1) {
            const gpu = gpus[0];
            if (gpu.util + gpuNumber > 1.001) {
                return null;
            }
            gpu.util += gpuNumber;
            this.trials.push({ gpuIndex: gpu.index, experimentId, trialId, util: gpuNumber });
            logger.debug(`Scheduled ${experimentId}/${trialId} -> ${gpu.index}`);
            return [ gpu.index ];

        } else {
            const n = Math.round(gpuNumber);
            if (gpus.length < n || gpus[n - 1].util > 0) {
                return null;
            }
            const indices = []
            for (const gpu of gpus.slice(0, n)) {
                gpu.util = 1;
                this.trials.push({ gpuIndex: gpu.index, experimentId, trialId, util: 1 });
                indices.push(gpu.index);
            }
            logger.debug(`Scheduled ${experimentId}/${trialId} ->`, indices);
            return indices.sort((a, b) => (a - b));
        }
    }

    public async release(experimentId: string, trialId: string): Promise<void> {
        this.releaseByFilter(trial => (trial.experimentId === experimentId && trial.trialId === trialId));
    }

    public async releaseAll(experimentId: string): Promise<void> {
        logger.info('Release whole experiment', experimentId);
        this.releaseByFilter(trial => (trial.experimentId === experimentId));
    }

    private updateGpus(info: GpuSystemInfo): void {
        for (const gpu of info.gpus) {
            const index = gpu.index;
            this.gpus[index].coreUtil = gpu.gpuCoreUtilization ?? 0;
            this.gpus[index].memUtil = gpu.gpuMemoryUtilization ?? 0;
            this.gpus[index].active = false;
            this.gpus[index].computeActive = false;
        }

        for (const proc of info.processes) {
            const index = proc.gpuIndex;
            this.gpus[index].active = true;
            if (proc.type === 'compute') {
                this.gpus[index].computeActive = true;
            }
        }
    }

    private sortGpus(restrict: ScheduleRestrictions): SchedulerGpuInfo[] {
        let gpus = this.gpus;
        if (restrict.onlyAcceptIndices) {
            gpus = gpus.filter(gpu => restrict.onlyAcceptIndices!.includes(gpu.index));
        }
        if (restrict.rejectActiveGpus) {
            gpus = gpus.filter(gpu => !gpu.active);
        }
        if (restrict.rejectComputeActiveGpus) {
            gpus = gpus.filter(gpu => !gpu.computeActive);
        }

        return gpus.sort((a, b) => {
            if (a.util !== b.util) {
                return a.util - b.util;
            }
            if (a.active !== b.active) {
                return Number(a.active) - Number(b.active);
            }
            if (a.computeActive !== b.computeActive) {
                return Number(a.computeActive) - Number(b.computeActive);
            }
            if (a.memUtil !== b.memUtil) {
                return a.memUtil - b.memUtil;
            }
            return a.coreUtil - b.coreUtil;
        });
    }

    private releaseByFilter(filter: (trial: SchedulerTrialInfo) => boolean): void {
        const trials = this.trials.filter(filter);
        trials.forEach(trial => {
            logger.debug(`Released ${trial.experimentId}/${trial.trialId}`);
            this.gpus[trial.gpuIndex].util -= trial.util;
        });
        this.trials = this.trials.filter(trial => !filter(trial));
    }
}
