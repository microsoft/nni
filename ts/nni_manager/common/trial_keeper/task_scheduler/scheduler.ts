// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  A simple GPU scheduler used by local and remote training services.
 **/

import { getLogger } from 'common/log';
import type { TrialKeeper } from 'common/trial_keeper/keeper';
import { GpuSystemInfo, collectGpuInfo as origCollectGpuInfo } from './collect_info';

const logger = getLogger('TaskScheduler');

let collectGpuInfo = origCollectGpuInfo;  // for ut

interface SchedulerGpuInfo {
    index: number;
    util: number;  // theoretical utilization calculated by NNI's trialGpuNumber
    coreUtil: number;  // real GPU core utilization (0 ~ 1)
    memUtil: number;  // real GPU memory utilization (0 ~ 1)
    active: boolean;
    computeActive: boolean;
}

interface SchedulerTrialInfo {
    gpuIndex: number;
    experimentId: string;
    trialId: string;
    util: number;
}

export class TaskScheduler {
    private gpus: SchedulerGpuInfo[] = [];
    private trials: SchedulerTrialInfo[] = [];

    /**
     *  Initialize the scheduler.
     *
     *  If there is no GPU found, throw error.
     **/
    public async init(): Promise<void> {
        const info = await collectGpuInfo(true);
        if (info === null) {
            throw new Error('TaskScheduler: Failed to collect GPU info');
        }
        if (info.gpuNumber === 0) {
            throw new Error('TaskScheduler: No GPU found');
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

    /**
     *  Update GPUs' utilization info.
     *
     *  If `force` is not true, it may use cached result.
     *
     *  This is normally unnecessary because it will implicitly update before scheduling.
     **/
    public async update(force?: boolean): Promise<void> {
        const info = await collectGpuInfo(force);
        if (info === null) {
            if (force) {
                throw new Error('TaskScheduler: Failed to update GPU info');
            }
            return;
        }
        if (info.gpuNumber !== this.gpus.length) {
            // TODO: according to yuge's experience this do might happen
            throw new Error(`TaskScheduler: GPU number changed from ${this.gpus.length} to ${info.gpuNumber}`);
        }

        this.updateGpus(info);
    }

    /**
     *  Schedule a trial and return its GPU-related environment variables.
     *
     *  If the trial cannot be scheduled, return `null`.
     *  (TODO: it might need more advanced failure handling)
     *
     *  This scheduler does NOT monitor the trial's life span.
     *  The caller must invokes `release()` or `releaseAll()` later.
     *
     *  `gpuNumber` can either be an integer or a float between 0 and 1.
     *
     *  The `restrictions` parameter may contain following options:
     *
     *  - onlyUseIndices:
     *
     *      Limit usable GPUs by index.
     *      This is `gpuIndices` in experiment config.
     *
     *  - rejectActive:
     *
     *      Do not use GPUs with running processes.
     *      This is reversed `useActiveGpu` in experiment config.
     *
     *  - rejectComputeActive:
     *
     *      Do not use GPUs with CUDA processes, but still use GPUs with graphics processes.
     *      This is useful for desktop systems with graphical interface.
     **/
    public async schedule(
            experimentId: string,
            trialId: string,
            gpuNumber: number,
            restrictions?: TrialKeeper.GpuRestrictions
        ): Promise<Record<string, string> | null> {

        if (gpuNumber === 0) {
            return { 'CUDA_VISIBLE_DEVICES': '' };
        }

        if (gpuNumber >= this.gpus.length) {
            // TODO: push this message to web portal
            logger.error(`Only have ${this.gpus.length} GPUs, requesting ${gpuNumber}`);
            return null;
        }

        const gpus = this.sortGpus(restrictions ?? {});

        if (gpuNumber < 1) {
            const gpu = gpus[0];
            if (gpu.util + gpuNumber > 1.001) {
                return null;
            }
            gpu.util += gpuNumber;
            this.trials.push({ gpuIndex: gpu.index, experimentId, trialId, util: gpuNumber });
            logger.debug(`Scheduled ${experimentId}/${trialId} -> ${gpu.index}`);
            return { 'CUDA_VISIBLE_DEVICES': String(gpu.index) };

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
            indices.sort((a, b) => (a - b));
            logger.debug(`Scheduled ${experimentId}/${trialId} ->`, indices);
            return { 'CUDA_VISIBLE_DEVICES': indices.join(',') };
        }
    }

    /**
     *  Release a trial's allocated GPUs.
     *
     *  If the trial does not exist, silently do nothing.
     **/
    public async release(experimentId: string, trialId: string): Promise<void> {
        this.releaseByFilter(trial => (trial.experimentId === experimentId && trial.trialId === trialId));
    }

    /**
     *  Release all trials of an experiment.
     *
     *  Useful when the experiment is shutting down or has lost response.
     **/
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

    private sortGpus(restrict: TrialKeeper.GpuRestrictions): SchedulerGpuInfo[] {
        let gpus = this.gpus.slice();  // copy for in-place sort
        if (restrict.onlyUseIndices) {
            gpus = gpus.filter(gpu => restrict.onlyUseIndices!.includes(gpu.index));
        }
        if (restrict.rejectActive) {
            gpus = gpus.filter(gpu => !gpu.active);
        }
        if (restrict.rejectComputeActive) {
            gpus = gpus.filter(gpu => !gpu.computeActive);
        }

        // prefer the gpu with lower theoretical utilization;
        // then the gpu without competing processes;
        // then the gpu with more free memory;
        // and finally the gpu with lower cuda core load.
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
            if (a.coreUtil !== b.coreUtil) {
                return a.coreUtil - b.coreUtil;
            }
            return a.index - b.index;
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

export namespace UnitTestHelpers {
    export function mockGpuInfo(info: GpuSystemInfo): void {
        collectGpuInfo = (_?: boolean): any => Promise.resolve(info);
    }

    export function getGpuUtils(scheduler: TaskScheduler): number[] {
        return (scheduler as any).gpus.map((gpu: SchedulerGpuInfo) => gpu.util);
    }
}
