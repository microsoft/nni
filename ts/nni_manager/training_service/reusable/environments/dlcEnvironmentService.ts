// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'fs';
import path from 'path';
import * as component from 'common/component';
import { getLogger, Logger } from 'common/log';
import { ExperimentConfig, DlcConfig, flattenConfig } from 'common/experimentConfig';
import { ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { DlcClient } from '../dlc/dlcClient';
import { DlcEnvironmentInformation } from '../dlc/dlcConfig';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { EventEmitter } from "events";
import { FileCommandChannel } from '../channels/fileCommandChannel';
import { MountedStorageService } from '../storages/mountedStorageService';
import { Scope } from 'typescript-ioc';
import { StorageService } from '../storageService';

interface FlattenDlcConfig extends ExperimentConfig, DlcConfig { }

/**
 * Collector DLC jobs info from DLC cluster, and update dlc job status locally
 */
@component.Singleton
export class DlcEnvironmentService extends EnvironmentService {

    private readonly log: Logger = getLogger('dlcEnvironmentService');
    private experimentId: string;
    private config: FlattenDlcConfig;

    constructor(config: ExperimentConfig, info: ExperimentStartupInfo) {
        super();
        this.experimentId = info.experimentId;
        this.config = flattenConfig(config, 'dlc');
        component.Container.bind(StorageService).to(MountedStorageService).scope(Scope.Singleton);
        const storageService = component.get<StorageService>(StorageService)
        const remoteRoot = storageService.joinPath(this.config.localStorageMountPoint, 'nni-experiments', this.experimentId);
        const localRoot = storageService.joinPath(this.config.localStorageMountPoint, 'nni-experiments');
        storageService.initialize(localRoot, remoteRoot);
    }

    public get hasStorageService(): boolean {
        return true;
    }

    public initCommandChannel(eventEmitter: EventEmitter): void {
        this.commandChannel = new FileCommandChannel(eventEmitter);
    }

    public createEnvironmentInformation(envId: string, envName: string): EnvironmentInformation {
        return new DlcEnvironmentInformation(envId, envName);
    }

    public get getName(): string {
        return 'dlc';
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        environments.forEach(async (environment) => {
            const dlcClient = (environment as DlcEnvironmentInformation).dlcClient;
            if (!dlcClient) {
                return Promise.reject('DLC client not initialized!');
            }
            const newStatus = await dlcClient.updateStatus(environment.status);
            switch (newStatus.toUpperCase()) {
                case 'CREATING':
                case 'CREATED':
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
                    return Promise.reject(`DLC: job ${environment.envId} is failed!`);
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
        const dlcEnvironment: DlcEnvironmentInformation = environment as DlcEnvironmentInformation;

        const environmentRoot = path.join(this.config.containerStorageMountPoint, `/nni-experiments/${this.experimentId}`);
        const localRoot = path.join(this.config.localStorageMountPoint, `/nni-experiments/${this.experimentId}`);

        dlcEnvironment.workingFolder = `${localRoot}/envs/${environment.id}`;
        dlcEnvironment.runnerWorkingFolder = `${environmentRoot}/envs/${environment.id}`;

        // environment id dir and command dir, folder created on DLC side can't be accessed on DSW.
        if (!fs.existsSync(`${dlcEnvironment.workingFolder}/commands`)) {
            await fs.promises.mkdir(`${dlcEnvironment.workingFolder}/commands`, {recursive: true});
        }

        environment.command = `cd ${environmentRoot} && ${environment.command} 1>${environment.runnerWorkingFolder}/trialrunner_stdout 2>${environment.runnerWorkingFolder}/trialrunner_stderr`;

        const dlcClient = new DlcClient(
            this.config.type,
            this.config.image,
            this.config.jobType,
            this.config.podCount,
            this.experimentId,
            environment.id,
            this.config.ecsSpec,
            this.config.region,
            this.config.nasDataSourceId,
            this.config.accessKeyId,
            this.config.accessKeySecret,
            environment.command,
        );

        dlcEnvironment.id = await dlcClient.submit();
        this.log.debug('dlc: before getTrackingUrl');
        dlcEnvironment.trackingUrl = await dlcClient.getTrackingUrl();
        this.log.debug(`dlc trackingUrl: ${dlcEnvironment.trackingUrl}`);
        dlcEnvironment.dlcClient = dlcClient;
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        const dlcEnvironment: DlcEnvironmentInformation = environment as DlcEnvironmentInformation;
        const dlcClient = dlcEnvironment.dlcClient;
        if (!dlcClient) {
            throw new Error('DLC client not initialized!');
        }
        dlcClient.stop();
    }
}
