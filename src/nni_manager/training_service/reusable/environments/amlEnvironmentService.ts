// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from "events";
import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import { getExperimentRootDir } from '../../../common/utils';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { validateCodeDir } from '../../common/util';
import { AMLClient } from '../aml/amlClient';
import { AMLClusterConfig, AMLEnvironmentInformation, AMLTrialConfig } from '../aml/amlConfig';
import { AMLCommandChannel } from '../channels/amlCommandChannel';
import { CommandChannel } from "../commandChannel";
import { EnvironmentInformation, EnvironmentService } from '../environment';


/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
@component.Singleton
export class AMLEnvironmentService extends EnvironmentService {

    private readonly log: Logger = getLogger();
    public amlClusterConfig: AMLClusterConfig | undefined;
    public amlTrialConfig: AMLTrialConfig | undefined;
    private experimentId: string;
    private experimentRootDir: string;

    constructor() {
        super();
        this.experimentId = getExperimentId();
        this.experimentRootDir = getExperimentRootDir();
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public createCommandChannel(commandEmitter: EventEmitter): CommandChannel {
        return new AMLCommandChannel(commandEmitter);
    }

    public createEnvironmentInformation(envId: string, envName: string): EnvironmentInformation {
        return new AMLEnvironmentInformation(envId, envName);
    }

    public async config(key: string, value: string): Promise<void> {
        switch (key) {
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
            default:
                this.log.debug(`AML not proccessed metadata key: '${key}', value: '${value}'`);
        }
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        environments.forEach(async (environment) => {
            const amlClient = (environment as AMLEnvironmentInformation).amlClient;
            if (!amlClient) {
                return Promise.reject('AML client not initialized!');
            }
            const newStatus = await amlClient.updateStatus(environment.status);
            switch (newStatus.toUpperCase()) {
                case 'WAITING':
                case 'QUEUED':
                    environment.setStatus('WAITING');
                    break;
                case 'RUNNING':
                    environment.setStatus('RUNNING');
                    break;
                case 'COMPLETED':
                case 'SUCCEEDED':
                    environment.setStatus('SUCCEEDED');
                    break;
                case 'FAILED':
                    environment.setStatus('FAILED');
                    return Promise.reject(`AML: job ${environment.envId} is failed!`);
                case 'STOPPED':
                case 'STOPPING':
                    environment.setStatus('USER_CANCELED');
                    break;
                default:
                    environment.setStatus('UNKNOWN');
            }
        });
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.amlClusterConfig === undefined) {
            throw new Error('AML Cluster config is not initialized');
        }
        if (this.amlTrialConfig === undefined) {
            throw new Error('AML trial config is not initialized');
        }
        const amlEnvironment: AMLEnvironmentInformation = environment as AMLEnvironmentInformation;
        const environmentLocalTempFolder = path.join(this.experimentRootDir, this.experimentId, "environment-temp");
        environment.command = `import os\nos.system('${amlEnvironment.command}')`;
        environment.useActiveGpu = this.amlClusterConfig.useActiveGpu;
        environment.maxTrialNumberPerGpu = this.amlClusterConfig.maxTrialNumPerGpu;
        await fs.promises.writeFile(path.join(environmentLocalTempFolder, 'nni_script.py'), amlEnvironment.command, { encoding: 'utf8' });
        const amlClient = new AMLClient(
            this.amlClusterConfig.subscriptionId,
            this.amlClusterConfig.resourceGroup,
            this.amlClusterConfig.workspaceName,
            this.experimentId,
            this.amlClusterConfig.computeTarget,
            this.amlTrialConfig.image,
            'nni_script.py',
            environmentLocalTempFolder
        );
        amlEnvironment.id = await amlClient.submit();
        amlEnvironment.trackingUrl = await amlClient.getTrackingUrl();
        amlEnvironment.amlClient = amlClient;
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        const amlEnvironment: AMLEnvironmentInformation = environment as AMLEnvironmentInformation;
        const amlClient = amlEnvironment.amlClient;
        if (!amlClient) {
            throw new Error('AML client not initialized!');
        }
        amlClient.stop();
    }
}
