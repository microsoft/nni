// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from "events";
import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../common/component';
import { getLogger, Logger } from '../../../common/log';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { HeterogenousCommandChannel } from '../channels/heterogeneousCommandChannel';
import { CommandChannel } from "../commandChannel";
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { AMLEnvironmentService } from './amlEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { randomSelect } from '../../../common/utils';
import { HeterogenousConfig } from '../heterogenous/heterogenousConfig';
import { WebCommandChannel } from '../channels/webCommandChannel';


/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
@component.Singleton
export class HeteroGeneousEnvironmentService extends EnvironmentService {
    
    private amlEnvironmentService: AMLEnvironmentService;
    private remoteEnvironmentService: RemoteEnvironmentService;
    private localEnvironmentService: LocalEnvironmentService;
    private paiEnvironmentService: OpenPaiEnvironmentService;
    private heterogenousConfig?: HeterogenousConfig;

    private readonly log: Logger = getLogger();

    constructor() {
        super();
        this.amlEnvironmentService = new AMLEnvironmentService();
        this.remoteEnvironmentService = new RemoteEnvironmentService();
        this.localEnvironmentService = new LocalEnvironmentService();
        this.paiEnvironmentService = new OpenPaiEnvironmentService();
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public createCommandChannel(commandEmitter: EventEmitter): CommandChannel {
        if (this.heterogenousConfig === undefined) {
            throw new Error('heterogenousConfig not initialized!');
        }
        return new HeterogenousCommandChannel(commandEmitter, this.heterogenousConfig.trainingServicePlatforms);
    }

    public async config(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.AML_CLUSTER_CONFIG:
                await this.amlEnvironmentService.config(key, value);
                break;
            case TrialConfigMetadataKey.MACHINE_LIST:
                await this.remoteEnvironmentService.config(key, value);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                await this.amlEnvironmentService.config(key, value);
                await this.remoteEnvironmentService.config(key, value);
                await this.paiEnvironmentService.config(key, value);
                await this.localEnvironmentService.config(key, value);
                break;
            case TrialConfigMetadataKey.PAI_CLUSTER_CONFIG:
                await this.paiEnvironmentService.config(key, value);
                break;
            case TrialConfigMetadataKey.LOCAL_CONFIG:
                await this.localEnvironmentService.config(key, value);
                break;
            case TrialConfigMetadataKey.HETEROGENEOUS_CONFIG:
                this.heterogenousConfig = <HeterogenousConfig>JSON.parse(value);
                break;
            default:
                this.log.debug(`Heterogenous not support metadata key: '${key}', value: '${value}'`);
        }
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        const tasks: Promise<void>[] = [];
        const openPaiEnvironments: EnvironmentInformation[] = [];
        environments.forEach(async (environment) => {
            switch (environment.platform) {
                case 'aml':
                    tasks.push(this.amlEnvironmentService.refreshEnvironment(environment));
                    break;
                case 'remote':
                    tasks.push(this.remoteEnvironmentService.refreshEnvironment(environment));
                    break;
                case 'local':
                    tasks.push(this.localEnvironmentService.refreshEnvironment(environment));
                    break;
                case 'pai':
                    openPaiEnvironments.push(environment);
                    break;
                default:
                    throw new Error(`Heterogenous not support platform: '${environment.platform}'`);
            }
        });
        // OpenPai only support refreshEnvironmentsStatus
        if (openPaiEnvironments.length) {
            tasks.push(this.paiEnvironmentService.refreshEnvironmentsStatus(openPaiEnvironments));
        }

        await Promise.all(tasks);
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.heterogenousConfig === undefined) {
            throw new Error('heterogenousConfig not initialized!');
        }
        console.log('traningServicePlatforms: ')
        console.log(this.heterogenousConfig.trainingServicePlatforms)
        const platform = randomSelect(this.heterogenousConfig.trainingServicePlatforms);
        switch (platform) {
            case 'aml':
                environment.platform = 'aml';
                await this.amlEnvironmentService.startEnvironment(environment);
                break;
            case 'remote':
                environment.platform = 'remote';
                await this.remoteEnvironmentService.startEnvironment(environment);
                break;
            case 'local':
                environment.platform = 'local';
                await this.localEnvironmentService.startEnvironment(environment);
                break;
            case 'pai':
                environment.platform = 'pai';
                await this.paiEnvironmentService.startEnvironment(environment);
                break;
        }
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        switch (environment.platform) {
            case 'aml':
                await this.amlEnvironmentService.stopEnvironment(environment);
                break;
            case 'remote':
                await this.remoteEnvironmentService.stopEnvironment(environment);
                break;
            case 'local':
                await this.localEnvironmentService.stopEnvironment(environment);
                break;
            case 'pai':
                await this.paiEnvironmentService.stopEnvironment(environment);
                break;
            default:
                throw new Error(`Heterogenous not support platform '${environment.platform}'`);
        }
    }
}
