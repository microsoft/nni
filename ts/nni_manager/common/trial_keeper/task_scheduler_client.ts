// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  FIXME: This is a placeholder.
 *
 *  In near future (before 3.0 release) GPU scheduler will become a shared daemon so
 *  multiple NNI experiments can gracefully share GPUs on one machine,
 *  and this class is the client to communicate with that daemon.
 *
 *  However as a step by step progress it currently embed a GPU scheduler instance inside the class.
 **/

import globals from 'common/globals';
import { getLogger } from 'common/log';
import type { TrialKeeper } from './keeper';
import { TaskScheduler } from './task_scheduler';

const logger = getLogger('TaskSchedulerClient');

export class TaskSchedulerClient {
    private server: TaskScheduler | null = null;

    constructor(enable: boolean) {
        if (enable) {
            this.server = new TaskScheduler();
        }
    }

    public async start(): Promise<void> {
        if (this.server !== null) {
            await this.server.init();
        }
    }

    public async shutdown(): Promise<void> {
        if (this.server !== null) {
            await this.server.releaseAll(globals.args.experimentId);
        }
    }

    public async schedule(trialId: string, gpuNumber?: number, restrictions?: TrialKeeper.GpuRestrictions):
            Promise<Record<string, string> | null> {

        if (gpuNumber === undefined) {
            return {};
        }
        if (gpuNumber === 0) {
            return { 'CUDA_VISIBLE_DEVICES': '' };
        }

        if (this.server === null) {
            logger.error(`GPU scheduling is not enabled, but gpuNumber of trial ${trialId} is ${gpuNumber}`);
            return null;
        }
        return this.server.schedule(globals.args.experimentId, trialId, gpuNumber, restrictions);
    }

    public async release(trialId: string): Promise<void> {
        if (this.server !== null) {
            await this.server.release(globals.args.experimentId, trialId);
        }
    }

    public onUtilityUpdate(callback: (info: Record<string, any>) => void): void {
        if (this.server !== null) {
            this.server.onUtilityUpdate(callback);
        }
    }
}
