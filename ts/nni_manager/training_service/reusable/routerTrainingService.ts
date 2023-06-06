// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { getLogger, Logger } from 'common/log';
import { MethodNotImplementedError } from 'common/errors';
import { ExperimentConfig, RemoteConfig, KubeflowConfig, FrameworkControllerConfig } from 'common/experimentConfig';
import { TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric } from 'common/trainingService';
import { delay } from 'common/utils';
import { KubeflowTrainingService } from '../kubernetes/kubeflow/kubeflowTrainingService';
import { FrameworkControllerTrainingService } from '../kubernetes/frameworkcontroller/frameworkcontrollerTrainingService';
import { TrialDispatcher } from './trialDispatcher';


/**
 * It's a intermedia implementation to support reusable training service.
 * The final goal is to support reusable training job in higher level than training service.
 */
class RouterTrainingService implements TrainingService {
    private log!: Logger;
    private internalTrainingService!: TrainingService;

    public static async construct(config: ExperimentConfig): Promise<RouterTrainingService> {
        const instance = new RouterTrainingService();
        instance.log = getLogger('RouterTrainingService');
        const platform = Array.isArray(config.trainingService) ? 'hybrid' : config.trainingService.platform;
        if (platform === 'remote' && (<RemoteConfig>config.trainingService).reuseMode === false) {
            throw new Error('Unexpected: non-reuse remote enters RouterTrainingService');
        } else if (platform === 'kubeflow' && (<KubeflowConfig>config.trainingService).reuseMode === false) {
            instance.internalTrainingService = new KubeflowTrainingService();
        } else if (platform === 'frameworkcontroller' && (<FrameworkControllerConfig>config.trainingService).reuseMode === false) {
            instance.internalTrainingService = new FrameworkControllerTrainingService();
        } else {
            instance.internalTrainingService = await TrialDispatcher.construct(config);
        }
        return instance;
    }

    // eslint-disable-next-line @typescript-eslint/no-empty-function
    private constructor() { }

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

    public async getTrialFile(_trialJobId: string, _fileName: string): Promise<string | Buffer> {
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

    public async getTrialOutputLocalPath(trialJobId: string): Promise<string> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return this.internalTrainingService.getTrialOutputLocalPath(trialJobId);
    }

    public async fetchTrialOutput(trialJobId: string, subpath: string): Promise<void> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return this.internalTrainingService.fetchTrialOutput(trialJobId, subpath);
    }
}

export { RouterTrainingService };
