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

import * as assert from 'assert';
import { EventEmitter } from 'events';
import { TrainingService, TrialJobDetail, TrialJobStatus } from '../common/trainingService';
import { delay } from '../common/utils';

type TrialJobMaintainerEvent = TrialJobStatus | 'EXPERIMENT_DONE' | 'NO_RUNNING_TRIALS';

/**
 * TrialJobs
 */
class TrialJobs {
    private eventEmitter: EventEmitter;
    private trialJobs: Map<string, TrialJobDetail>;
    private noMoreTrials: boolean;
    private reachedMaxTrialNum: boolean;
    private stopLoop: boolean;
    private trainingService: TrainingService;
    private pastExecDuration: number; // second
    private maxExecDuration: number; // second
    private isRunning: boolean;

    constructor(
        trainingService: TrainingService,
        pastExecDuration: number, // second
        maxExecDuration: number // second
    ) {
        this.eventEmitter = new EventEmitter();
        this.trialJobs = new Map<string, TrialJobDetail>();
        this.noMoreTrials = false;
        this.reachedMaxTrialNum = false;
        this.stopLoop = false;
        this.trainingService = trainingService;
        this.pastExecDuration = pastExecDuration;
        this.maxExecDuration = maxExecDuration;
        this.isRunning = true;
    }

    public isTrialJobsRunning(): boolean {
        return this.isRunning;
    }

    public setTrialJob(key: string, value: TrialJobDetail): void {
        this.trialJobs.set(key, value);
    }

    public getTrialJob(key: string): TrialJobDetail | undefined {
        return this.trialJobs.get(key);
    }

    public setReachMaxTrialNum(fact: boolean): void {
        this.reachedMaxTrialNum = fact;
    }

    public setNoMoreTrials(): void {
        // NOTE: this variable is not used, because even tuner has no more trials,
        // user could also submit customized trial jobs, thus the experiment should not stop.
        // that is, noMoreTrials does not control experiment's status.
        this.noMoreTrials = true;
    }

    public setStopLoop(): void {
        this.stopLoop = true;
    }

    public setMaxExecDuration(duration: number): void {
        this.maxExecDuration = duration;
    }

    public on(listener: (event: TrialJobMaintainerEvent, trialJobDetail: TrialJobDetail) => void): void {
        this.eventEmitter.addListener('all', listener);
    }

    public async requestTrialJobsStatus(): Promise<void> {
        if (this.trialJobs.size === 0) {
            // TODO: we can relax this condition, to reach the full concurrency in some corner cases.
            this.eventEmitter.emit('all', 'NO_RUNNING_TRIALS', undefined);
            return Promise.resolve();
        }
        for (const trialJobId of Array.from(this.trialJobs.keys())) {
            const trialJobDetail: TrialJobDetail = await this.trainingService.getTrialJob(trialJobId);
            switch (trialJobDetail.status) {
                case 'SUCCEEDED':
                case 'USER_CANCELED':
                    this.eventEmitter.emit('all', trialJobDetail.status, trialJobDetail);
                    this.trialJobs.delete(trialJobId);
                    break;
                case 'FAILED':
                case 'SYS_CANCELED':
                    // In the current version, we do not retry
                    // TO DO: push this job to queue for retry
                    this.eventEmitter.emit('all', trialJobDetail.status, trialJobDetail);
                    this.trialJobs.delete(trialJobId);
                    break;
                case 'WAITING':
                    // Do nothing
                    break;
                case 'RUNNING':
                    const oldTrialJobDetail: TrialJobDetail | undefined = this.trialJobs.get(trialJobId);
                    assert(oldTrialJobDetail);
                    if (oldTrialJobDetail !== undefined && oldTrialJobDetail.status === "WAITING") {
                        this.trialJobs.set(trialJobId, trialJobDetail);
                        this.eventEmitter.emit('all', trialJobDetail.status, trialJobDetail);
                    }
                    break;
                case 'UNKNOWN':
                    // Do nothing
                    break;
                default:
                // TO DO: add warning in log
            }
        }

        return Promise.resolve();
    }

    public async run(): Promise<void> {
        let startTime: number = Date.now();
        let pastExecDurationThisRun: number = 0;
        while (true) {
            if (this.stopLoop) {
                break;
            }
            if ((Date.now() - startTime) / 1000
                + pastExecDurationThisRun
                + this.pastExecDuration < this.maxExecDuration
                && !this.reachedMaxTrialNum) {
                if (!this.isRunning) {
                    startTime = Date.now();
                    this.isRunning = true;
                }
                await this.requestTrialJobsStatus();
            } else {
                this.isRunning = false;
                pastExecDurationThisRun += (Date.now() - startTime) / 1000;
            }
            await delay(5000);
        }
        this.eventEmitter.emit('all', 'EXPERIMENT_DONE');
    }
}

export { TrialJobs, TrialJobMaintainerEvent };
