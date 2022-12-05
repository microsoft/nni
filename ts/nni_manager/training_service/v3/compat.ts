// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EventEmitter } from 'events';
import { readFile } from 'fs/promises';
import path from 'path';

import { Deferred } from 'common/deferred';
import type { TrainingServiceConfig } from 'common/experimentConfig';
import globals from 'common/globals';
import {
    TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, TrialJobStatus
} from 'common/trainingService';
import type { EnvironmentInfo, Parameter, TrainingServiceV3 } from 'common/training_service_v3';
import { trainingServiceFactoryV3 } from './factory';

type MutableTrialJobDetail = {
    -readonly [Property in keyof TrialJobDetail]: TrialJobDetail[Property];
};

export class V3asV1 implements TrainingService {
    private config: TrainingServiceConfig;
    private v3: TrainingServiceV3;

    private emitter: EventEmitter = new EventEmitter();
    private runDeferred: Deferred<void> = new Deferred();
    private startDeferred: Deferred<void> = new Deferred();

    private trialJobs: Record<string, MutableTrialJobDetail> = {};
    private parameters: Record<string, Parameter> = {};

    private environments: EnvironmentInfo[] = [];
    private lastEnvId: string = '';

    constructor(config: TrainingServiceConfig) {
        this.config = config;
        this.v3 = trainingServiceFactoryV3(config);
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        return Object.values(this.trialJobs);
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        return this.trialJobs[trialJobId];
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.emitter.addListener('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.emitter.removeListener('metric', listener);
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        await this.startDeferred.promise;
        let trialId: string | null = null;
        while (trialId === null) {
            const envId = this.schedule();
            trialId = await this.v3.createTrial(envId, this.config.trialCommand, 'trial_code');
        }

        // In new interface, hyper parameters will be sent on demand.
        this.parameters[trialId] = form.hyperParameters.value;

        this.trialJobs[trialId] = {
            id: trialId,
            status: 'WAITING',
            submitTime: Date.now(),
            workingDirectory: '_unset_',  // never set in current remote training service, so it's optional
            form: form,
        };
        return this.trialJobs[trialId];
    }

    public async updateTrialJob(_trialJobId: string, _form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        // Seems never used.
        throw new Error('Not implemented: V3asV1.updateTrialJob()');
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean): Promise<void> {
        await this.v3.stopTrial(trialJobId);
        this.trialJobs[trialJobId].isEarlyStopped = Boolean(isEarlyStopped);
    }

    public async getTrialFile(trialJobId: string, fileName: string): Promise<Buffer | string> {
        const dir = path.join(globals.paths.experimentRoot, 'trials', trialJobId);

        let logPath: string | null = null;
        if (fileName === 'trial.log') {
            logPath = path.join(dir, 'trial.log');
        } else if (fileName === 'stderr') {
            logPath = path.join(dir, 'trial.stderr');
        } else if (fileName === 'stdout') {
            logPath = path.join(dir, 'trial.stdout');
        }

        if (logPath !== null) {
            return await readFile(logPath, { encoding: 'utf8' });
        } else {
            // FIXME
            // Need to fix `model.onnx`.
            // I guess it should be put inside `nni_outputs`.
            return await readFile(path.join(dir, 'output', fileName));
        }
    }

    public async setClusterMetadata(_key: string, _value: string): Promise<void> {
        throw new Error('Not implemented: V3asV1.setClusterMetadata()');
    }

    public async getClusterMetadata(_key: string): Promise<string> {
        throw new Error('Not implemented: V3asV1.getClusterMetadata()');
    }

    public async getTrialOutputLocalPath(trialJobId: string): Promise<string> {
        return path.join(globals.paths.experimentRoot, 'trials', trialJobId, 'output');
    }

    public async fetchTrialOutput(_trialJobId: string, _subpath: string): Promise<void> {
        // It is automatic.
    }

    public async cleanUp(): Promise<void> {
        await this.v3.stop();
        this.runDeferred.resolve();
    }

    public run(): Promise<void> {
        this.start();
        return this.runDeferred.promise;
    }

    private async start(): Promise<void> {
        await this.v3.init();

        this.v3.onRequestParameter(async (trialId) => {
            await this.v3.sendParameter(trialId, this.parameters[trialId]);
        });
        this.v3.onMetric(async (trialId, metric) => {
            this.emitter.emit('metric', { id: trialId, data: metric });
        });
        this.v3.onTrialStart(async (trialId, timestamp) => {
            this.trialJobs[trialId].status = 'RUNNING';
            this.trialJobs[trialId].startTime = timestamp;
        });
        this.v3.onTrialEnd(async (trialId, timestamp, exitCode) => {
            const trial = this.trialJobs[trialId];
            if (exitCode === 0) {
                trial.status = 'SUCCEEDED';
            } else if (exitCode !== null) {
                trial.status = 'FAILED';
            } else if (trial.isEarlyStopped) {
                trial.status = 'EARLY_STOPPED';
            } else {
                trial.status = 'USER_CANCELED';
            }
            trial.endTime = timestamp;
        });
        this.v3.onEnvironmentUpdate(async (environments) => {
            this.environments = environments;
        });

        this.environments = await this.v3.start();
        await this.v3.uploadDirectory('trial_code', this.config.trialCodeDirectory);

        this.startDeferred.resolve();
    }

    private schedule(): string {
        // Simple round-robin schedule.
        // Find the last used environment and select next one.
        // If the last used environment is not found (destroyed), use first environment.
        const prevIndex = this.environments.findIndex((env) => env.id === this.lastEnvId);
        const index = (prevIndex + 1) % this.environments.length;
        this.lastEnvId = this.environments[index].id;
        return this.lastEnvId;
    }
}
