// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from "events";
import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../common/component';
import { getLogger, Logger } from '../../../common/log';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { HeterogenousCommandChannel } from '../channels/heterogenousCommandChannel';
import { CommandChannel } from "../commandChannel";
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { AMLEnvironmentService } from './amlEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { randomSelect } from '../../../common/utils';


/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
@component.Singleton
export class HeteroGenousEnvironmentService extends EnvironmentService {
    
    private amlEnvironmentService: AMLEnvironmentService;
    private remoteEnvironmentService: RemoteEnvironmentService;

    private readonly log: Logger = getLogger();

    constructor() {
        super();
        this.amlEnvironmentService = new AMLEnvironmentService();
        this.remoteEnvironmentService = new RemoteEnvironmentService();
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public createCommandChannel(commandEmitter: EventEmitter): CommandChannel {
        return new HeterogenousCommandChannel(commandEmitter);
    }

    public async config(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.AML_CLUSTER_CONFIG:
                this.amlEnvironmentService.config(key, value);
                break;
            case TrialConfigMetadataKey.MACHINE_LIST:
                this.remoteEnvironmentService.config(key, value);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG: 
                this.amlEnvironmentService.config(key, value);
                this.remoteEnvironmentService.config(key, value);     
            default:
                this.log.debug(`Heterogenous not support metadata key: '${key}', value: '${value}'`);
        }
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        const tasks: Promise<void>[] = [];
        environments.forEach(async (environment) => {
            switch (environment.platform) {
                case 'aml':
                    tasks.push(this.amlEnvironmentService.refreshEnvironment(environment));
                    break;
                case 'remote':
                    tasks.push(this.remoteEnvironmentService.refreshEnvironment(environment));
                    break;
                default:
                    throw new Error(`Heterogenous not support platform: '${environment.platform}'`);
            }
        });
        await Promise.all(tasks);
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        const number = randomSelect([0, 1]);
        switch (number) {
            case 0:
                environment.platform = 'aml';
                this.amlEnvironmentService.startEnvironment(environment);
                break;
            case 1:
                environment.platform = 'remote';
                this.remoteEnvironmentService.startEnvironment(environment);
                break;
        }
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        switch (environment.platform) {
            case 'aml':
                this.amlEnvironmentService.stopEnvironment(environment);
                break;
            case 'remote':
                this.remoteEnvironmentService.stopEnvironment(environment);
                break;
            default:
                throw new Error(`Heterogenous not support platform '${environment.platform}'`);
        }
    }
}
