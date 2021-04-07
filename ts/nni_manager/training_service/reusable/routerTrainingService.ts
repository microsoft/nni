// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as component from '../../common/component';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors';
import { ExperimentConfig, RemoteConfig, OpenpaiConfig } from '../../common/experimentConfig';
import { TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, LogType } from '../../common/trainingService';
import { delay } from '../../common/utils';
import { PAITrainingService } from '../pai/paiTrainingService';
import { RemoteMachineTrainingService } from '../remote_machine/remoteMachineTrainingService';
import { TrialDispatcher } from './trialDispatcher';


/**
 * It's a intermedia implementation to support reusable training service.
 * The final goal is to support reusable training job in higher level than training service.
 */
@component.Singleton
class RouterTrainingService implements TrainingService {
    protected readonly log: Logger;
    private internalTrainingService: TrainingService;

    constructor(config: ExperimentConfig) {
        this.log = getLogger();
        const platform = Array.isArray(config.trainingService) ? 'hybrid' : config.trainingService.platform;
        if (platform === 'remote' && !(<RemoteConfig>config.trainingService).reuseMode) {
            this.internalTrainingService = new RemoteMachineTrainingService(config);
        } else if (platform === 'openpai' && !(<OpenpaiConfig>config.trainingService).reuseMode) {
            this.internalTrainingService = new PAITrainingService(config);
        } else {
            this.internalTrainingService = new TrialDispatcher(config);
        }
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.listTrialJobs();
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.getTrialJob(trialJobId);
    }

    public async getTrialLog(_trialJobId: string, _logType: LogType): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        this.internalTrainingService.addTrialJobMetricListener(listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        this.internalTrainingService.removeTrialJobMetricListener(listener);
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.submitTrialJob(form);
    }

    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.updateTrialJob(trialJobId, form);
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean | undefined): Promise<void> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        await this.internalTrainingService.cancelTrialJob(trialJobId, isEarlyStopped);
    }

    public async setClusterMetadata(_key: string, _value: string): Promise<void> { return; }
    public async getClusterMetadata(_key: string): Promise<string> { return ''; }

    public async cleanUp(): Promise<void> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        await this.internalTrainingService.cleanUp();
    }

    public async run(): Promise<void> {
        // wait internal training service is assigned.
        // It will be assigned after set metadata of paiConfig
        while (this.internalTrainingService === undefined) {
            await delay(100);
        }
        return await this.internalTrainingService.run();
    }
}

export { RouterTrainingService };
