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

import * as component from '../../common/component';
import { getLogger, Logger } from '../../common/log';
import { TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric } from '../../common/trainingService';
import { delay } from '../../common/utils';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { PAIClusterConfig } from '../pai/paiConfig';
import { AMLClusterConfig } from '../aml/amlConfig';
import { PAIK8STrainingService } from '../pai/paiK8S/paiK8STrainingService';
import { AMLTrainingService } from '../aml/amlTrainingService';
import { TrialDispatcher } from './trialDispatcher';
import { Container, Scope } from 'typescript-ioc';
import { EnvironmentService } from './environment';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { AMLEnvironmentService } from './amlEnvironmentService';
import { StorageService } from './storageService';
import { MountedStorageService } from './mountedStorageService';
import { TrialService } from './trial';
import { StorageTrialService } from './storageTrialService';


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
                    // TODO to support other trialService  later.
                    Container.bind(TrialService)
                        .to(StorageTrialService)
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
                const config = <AMLClusterConfig>JSON.parse(value);
                this.internalTrainingService = component.get(TrialDispatcher);

                Container.bind(EnvironmentService)
                    .to(AMLEnvironmentService)
                    .scope(Scope.Singleton);
                Container.bind(StorageService)
                    .to(MountedStorageService)
                    .scope(Scope.Singleton);
                Container.bind(TrialService)
                    .to(StorageTrialService)
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
