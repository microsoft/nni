// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'fs';
import path from 'path';
import { AmlConfig } from 'common/experimentConfig';
import { ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { IocShim } from 'common/ioc_shim';
import { getLogger, Logger } from 'common/log';
import { validateCodeDir } from 'training_service/common/util';
import { AMLClient } from '../aml/amlClient';
import { AMLEnvironmentInformation } from '../aml/amlConfig';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { EventEmitter } from "events";
import { AMLCommandChannel } from '../channels/amlCommandChannel';
import { SharedStorageService } from '../sharedStorage'

/**
 * Collector AML jobs info from AML cluster, and update aml job status locally
 */
export class AMLEnvironmentService extends EnvironmentService {

    private readonly log: Logger = getLogger('AMLEnvironmentService');
    private experimentId: string;
    private experimentRootDir: string;
    private config: AmlConfig;

    constructor(config: AmlConfig, info: ExperimentStartupInfo) {
        super();
        this.experimentId = info.experimentId;
        this.experimentRootDir = info.logDir;
        this.config = config;
        validateCodeDir(this.config.trialCodeDirectory);
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public initCommandChannel(eventEmitter: EventEmitter): void {
        this.commandChannel = new AMLCommandChannel(eventEmitter);
    }

    public createEnvironmentInformation(envId: string, envName: string): EnvironmentInformation {
        return new AMLEnvironmentInformation(envId, envName);
    }

    public get getName(): string {
        return 'aml';
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
        const amlEnvironment: AMLEnvironmentInformation = environment as AMLEnvironmentInformation;
        const environmentLocalTempFolder = path.join(this.experimentRootDir, "environment-temp");
        if (!fs.existsSync(environmentLocalTempFolder)) {
            await fs.promises.mkdir(environmentLocalTempFolder, {recursive: true});
        }
        if (amlEnvironment.useSharedStorage) {
            const environmentRoot = IocShim.get<SharedStorageService>(SharedStorageService).remoteWorkingRoot;
            const remoteMountCommand = IocShim.get<SharedStorageService>(SharedStorageService).remoteMountCommand;
            amlEnvironment.command = `${remoteMountCommand} && cd ${environmentRoot} && ${amlEnvironment.command}`.replace(/"/g, `\\"`);
        } else {
            amlEnvironment.command = `mv envs outputs/envs && cd outputs && ${amlEnvironment.command}`;
        }
        amlEnvironment.command = `import os\nos.system('${amlEnvironment.command}')`;
        amlEnvironment.maxTrialNumberPerGpu = this.config.maxTrialNumberPerGpu;

        await fs.promises.writeFile(path.join(environmentLocalTempFolder, 'nni_script.py'), amlEnvironment.command, { encoding: 'utf8' });
        const amlClient = new AMLClient(
            this.config.subscriptionId,
            this.config.resourceGroup,
            this.config.workspaceName,
            this.experimentId,
            this.config.computeTarget,
            this.config.dockerImage,
            'nni_script.py',
            environmentLocalTempFolder
        );
        this.log.debug('aml: before amlClient submit');
        amlEnvironment.id = await amlClient.submit();
        this.log.debug('aml: before getTrackingUrl');
        amlEnvironment.trackingUrl = await amlClient.getTrackingUrl();
        this.log.debug('aml: after getTrackingUrl');
        amlEnvironment.amlClient = amlClient;
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        const amlEnvironment: AMLEnvironmentInformation = environment as AMLEnvironmentInformation;
        const amlClient = amlEnvironment.amlClient;
        if (!amlClient) {
            throw new Error('AML client not initialized!');
        }
        const result = await amlClient.stop();
        if (result) {
            this.log.info(`Stop aml run ${environment.id} success!`);
        } else {
            this.log.info(`Stop aml run ${environment.id} failed!`);
        }
    }
}
