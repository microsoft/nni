// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as path from 'path';
import { Deferred } from 'ts-deferred';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { AMLClusterConfig, AMLTrialConfig } from '../aml/amlConfig';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { AMLEnvironmentInformation } from '../aml/amlConfig';
import { AMLClient } from '../aml/amlClient';
import {
    NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../../common/trainingService';
import { execMkdir, validateCodeDir, execCopydir } from '../../common/util';
import {
    delay, generateParamFileName, getExperimentRootDir, getIPV4Address, getJobCancelStatus,
    getVersion, uniqueString
} from '../../../common/utils';
import { AMLCommandChannel } from '../channels/amlCommandChannel';
import { CommandChannel } from "../commandChannel";
import { EventEmitter } from "events";


/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
@component.Singleton
export class AMLEnvironmentService extends EnvironmentService {
    
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
        super();
        this.experimentId = getExperimentId();
        this.experimentRootDir = getExperimentRootDir();
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public getCommandChannel(commandEmitter: EventEmitter): CommandChannel {
        return new AMLCommandChannel(commandEmitter);
    }

    public createEnviornmentInfomation(envId: string, envName: string): EnvironmentInformation {
        return new AMLEnvironmentInformation(envId, envName);
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
            let amlClient = (environment as AMLEnvironmentInformation).amlClient;
                    if (!amlClient) {
            throw new Error('AML client not initialized!');
            }
            let status = await amlClient.updateStatus(environment.status);
            switch (status.toUpperCase()) {
                case 'WAITING':
                case 'RUNNING':
                case 'QUEUED':
                    break;
                case 'COMPLETED':
                    environment.status = 'SUCCEEDED';
                case 'SUCCEEDED':
                    environment.status = 'SUCCEEDED';
                    break;
                case 'FAILED':
                    environment.status = 'FAILED';
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
        let amlEnvironment: AMLEnvironmentInformation = environment as AMLEnvironmentInformation;
        let environmentLocalTempFolder = path.join(this.experimentRootDir, this.experimentId, "environment-temp");
        environment.command = `import os\nos.system('${amlEnvironment.command}')`;
        await fs.promises.writeFile(path.join(environmentLocalTempFolder, 'nni_script.py'), amlEnvironment.command ,{ encoding: 'utf8' });
        let amlClient = new AMLClient(
            this.amlClusterConfig.subscriptionId,
            this.amlClusterConfig.resourceGroup,
            this.amlClusterConfig.workspaceName,
            this.experimentId,
            this.amlTrialConfig.computerTarget,
            this.amlTrialConfig.nodeCount,
            this.amlTrialConfig.image,
            'nni_script.py',
            environmentLocalTempFolder
        );
        amlEnvironment.id = await amlClient.submit();
        amlEnvironment.trackingUrl = await amlClient.getTrackingUrl();
        amlEnvironment.amlClient = amlClient;
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        let amlEnvironment: AMLEnvironmentInformation = environment as AMLEnvironmentInformation;
        let amlClient = amlEnvironment.amlClient;
        if (!amlClient) {
            throw new Error('AML client not initialized!');
        }
        amlClient.stop();
    }
}
