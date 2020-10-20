// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Container, Scope } from 'typescript-ioc';
import * as component from '../../common/component';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors'
import { TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, LogType } from '../../common/trainingService';
import { delay } from '../../common/utils';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { PAIClusterConfig } from '../pai/paiConfig';
import { PAIK8STrainingService } from '../pai/paiK8S/paiK8STrainingService';
import { RemoteMachineTrainingService } from '../remote_machine/remoteMachineTrainingService';
import { EnvironmentService } from './environment';
import { OpenPaiEnvironmentService } from './environments/openPaiEnvironmentService';
import { AMLEnvironmentService } from './environments/amlEnvironmentService';
import { RemoteEnvironmentService } from './environments/remoteEnvironmentService';
import { MountedStorageService } from './storages/mountedStorageService';
import { StorageService } from './storageService';
import { TrialDispatcher } from './trialDispatcher';
import { RemoteConfig } from './remote/remoteConfig';


/**
 * It's a intermedia implementation to support reusable training service.
 * The final goal is to support reusable training job in higher level than training service.
 */
@component.Singleton
class RouterTrainingService implements TrainingService {
    protected readonly log!: Logger;
    private internalTrainingService: TrainingService | undefined;
    private metaDataCache: Map<string, string> = new Map<string, string>();

    constructor() {
        this.log = getLogger();
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

    public get isMultiPhaseJobSupported(): boolean {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return this.internalTrainingService.isMultiPhaseJobSupported;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean | undefined): Promise<void> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        await this.internalTrainingService.cancelTrialJob(trialJobId, isEarlyStopped);
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        if (this.internalTrainingService === undefined) {
            if (key === TrialConfigMetadataKey.PAI_CLUSTER_CONFIG) {
                const config = <PAIClusterConfig>JSON.parse(value);
                if (config.reuse === true) {
                    this.log.info(`reuse flag enabled, use EnvironmentManager.`);
                    this.internalTrainingService = component.get(TrialDispatcher);

                    // TODO to support other serivces later.
                    Container.bind(EnvironmentService)
                        .to(OpenPaiEnvironmentService)
                        .scope(Scope.Singleton);
                    // TODO to support other storages later.
                    Container.bind(StorageService)
                        .to(MountedStorageService)
                        .scope(Scope.Singleton);
                } else {
                    this.log.debug(`caching metadata key:{} value:{}, as training service is not determined.`);
                    this.internalTrainingService = component.get(PAIK8STrainingService);
                }
                for (const [key, value] of this.metaDataCache) {
                    if (this.internalTrainingService === undefined) {
                        throw new Error("TrainingService is not assigned!");
                    }
                    await this.internalTrainingService.setClusterMetadata(key, value);
                }

                if (this.internalTrainingService === undefined) {
                    throw new Error("TrainingService is not assigned!");
                }
                await this.internalTrainingService.setClusterMetadata(key, value);

                this.metaDataCache.clear();
            } else if (key === TrialConfigMetadataKey.AML_CLUSTER_CONFIG) {
                this.internalTrainingService = component.get(TrialDispatcher);

                Container.bind(EnvironmentService)
                    .to(AMLEnvironmentService)
                    .scope(Scope.Singleton);
                for (const [key, value] of this.metaDataCache) {
                    if (this.internalTrainingService === undefined) {
                        throw new Error("TrainingService is not assigned!");
                    }
                    await this.internalTrainingService.setClusterMetadata(key, value);
                }

                if (this.internalTrainingService === undefined) {
                    throw new Error("TrainingService is not assigned!");
                }
                await this.internalTrainingService.setClusterMetadata(key, value);

                this.metaDataCache.clear();
            } else if (key === TrialConfigMetadataKey.REMOTE_CONFIG) {
                const config = <RemoteConfig>JSON.parse(value);
                if (config.reuse === true) {
                    this.log.info(`reuse flag enabled, use EnvironmentManager.`);
                    this.internalTrainingService = component.get(TrialDispatcher);
                    Container.bind(EnvironmentService)
                        .to(RemoteEnvironmentService)
                        .scope(Scope.Singleton);
                } else {
                    this.log.debug(`caching metadata key:{} value:{}, as training service is not determined.`);
                    this.internalTrainingService = component.get(RemoteMachineTrainingService);
                }
            } else {
                this.log.debug(`caching metadata key:{} value:{}, as training service is not determined.`);
                this.metaDataCache.set(key, value);
            }
        } else {
            await this.internalTrainingService.setClusterMetadata(key, value);
        }
    }

    public async getClusterMetadata(key: string): Promise<string> {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.getClusterMetadata(key);
    }

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
