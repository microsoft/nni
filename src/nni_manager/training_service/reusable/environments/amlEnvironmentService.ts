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

import * as fs from 'fs';
import * as path from 'path';
import { Deferred } from 'ts-deferred';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { AMLClusterConfig, AMLTrialConfig, AMLTrialJobDetail } from '../../aml/amlConfig';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { AMLClient } from '../aml/amlData';
import {
    NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../../common/trainingService';
import { execMkdir, validateCodeDir, execCopydir } from '../../common/util';
import {
    delay, generateParamFileName, getExperimentRootDir, getIPV4Address, getJobCancelStatus,
    getVersion, uniqueString
} from '../../../common/utils';

/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
@component.Singleton
export class AMLEnvironmentService implements EnvironmentService {
    
    private readonly log: Logger = getLogger();
    public amlClusterConfig: AMLClusterConfig | undefined;
    public amlTrialConfig: AMLTrialConfig | undefined;
    private amlJobConfig: any;
    private stopping: boolean = false;
    private versionCheck: boolean = true;
    private isMultiPhase: boolean = false;
    private nniVersion?: string;
    private experimentId: string;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    private experimentRootDir: string;

    constructor() {
        this.experimentId = getExperimentId();
        this.experimentRootDir = getExperimentRootDir();
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public async config(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.AML_CLUSTER_CONFIG:
                this.amlClusterConfig = <AMLClusterConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                if (this.amlClusterConfig === undefined) {
                    this.log.error('aml cluster config is not initialized');
                    break;
                }
                this.amlTrialConfig = <AMLTrialConfig>JSON.parse(value);
                // Validate to make sure codeDir doesn't have too many files
                await validateCodeDir(this.amlTrialConfig.codeDir);
                break;
            }
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                this.nniVersion = this.versionCheck ? await getVersion() : '';
                break;
            case TrialConfigMetadataKey.MULTI_PHASE:
                this.isMultiPhase = (value === 'true' || value === 'True');
                break;
            default:
                //Reject for unknown keys
                this.log.error(`Uknown key: ${key}`);
        }
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        environments.forEach(async (environment) => {
            let amlClient = environment.environmentClient;
            let status = await amlClient.updateStatus(environment.status);
            switch (status.toUpperCase()) {
                case 'QUEUED':
                    environment.status = 'WAITING';
                    break;
                case 'COMPLETED':
                    environment.status = 'SUCCEEDED';
                    break;
                case 'WAITING':
                case 'RUNNING':
                case 'SUCCEEDED':
                case 'FAILED':
                    environment.status = status.toUpperCase();
                    break;
                case 'STOPPED':
                case 'STOPPING':
                    environment.status = 'USER_CANCELED';
                    break;
                default:
                    environment.status = 'UNKNOWN';
            }
        });
        deferred.resolve();
        return deferred.promise;
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.amlClusterConfig === undefined) {
            throw new Error('AML Cluster config is not initialized');
        }
        if (this.amlTrialConfig === undefined) {
            throw new Error('AML trial config is not initialized');
        }
        await fs.promises.writeFile(path.join(environment.environmentLocalTempFolder, 'nni_script.py'), environment.command ,{ encoding: 'utf8' });
        let amlClient = new AMLClient(
            this.amlClusterConfig.subscriptionId,
            this.amlClusterConfig.resourceGroup,
            this.amlClusterConfig.workspaceName,
            this.experimentId,
            this.amlTrialConfig.computerTarget,
            this.amlTrialConfig.image,
            'nni_script.py',
            environment.environmentLocalTempFolder
        );
        environment.id = await amlClient.submit();
        environment.trackingUrl = await amlClient.getTrackingUrl();
        environment.environmentClient = amlClient;
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        let amlClient = environment.environmentClient;
        amlClient.stop();
    }
}
