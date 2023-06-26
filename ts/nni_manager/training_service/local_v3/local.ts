// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import path from 'node:path';

import { globals } from 'common/globals';
import { Logger, getLogger } from 'common/log';
import type { LocalConfig, TrainingServiceConfig } from 'common/experimentConfig';
import type { EnvironmentInfo, Metric, Parameter, TrainingServiceV3 } from 'common/training_service_v3';
import { TrialKeeper } from 'common/trial_keeper/keeper';

export class LocalTrainingServiceV3 implements TrainingServiceV3 {
    private config: LocalConfig;
    private env: EnvironmentInfo;
    private log: Logger;
    private trialKeeper: TrialKeeper;

    constructor(trainingServiceId: string, config: TrainingServiceConfig) {
        this.log = getLogger(`LocalV3.${trainingServiceId}`);
        this.log.debug('Training sevice config:', config);

        this.config = config as LocalConfig;
        this.env = { id: `${trainingServiceId}-env` };
        this.trialKeeper = new TrialKeeper(this.env.id, 'local', Boolean(config.trialGpuNumber));
    }

    public async init(): Promise<void> {
        return;
    }

    public async start(): Promise<EnvironmentInfo[]> {
        this.log.info('Start');
        await this.trialKeeper.start();
        return [ this.env ];
    }

    public async stop(): Promise<void> {
        await this.trialKeeper.shutdown();
        this.log.info('All trials stopped');
    }

    /**
     *  Note:
     *  The directory is not copied, so changes in code directory will affect new trials.
     *  This is different from all other training services.
     **/
    public async uploadDirectory(directoryName: string, path: string): Promise<void> {
        this.log.info(`Register directory ${directoryName} = ${path}`);
        this.trialKeeper.registerDirectory(directoryName, path);
    }

    public async createTrial(
        _envId: string,
        trialCommand: string,
        directoryName: string,
        sequenceId: number,
        trialId?: string
    ): Promise<string | null> {

        trialId = trialId ?? uuid();

        let gpuNumber = this.config.trialGpuNumber;
        if (gpuNumber) {
            gpuNumber /= this.config.maxTrialNumberPerGpu;
        }

        const opts: TrialKeeper.TrialOptions = {
            id: trialId,
            command: trialCommand,
            codeDirectoryName: directoryName,
            sequenceId,
            gpuNumber,
            gpuRestrictions: {
                onlyUseIndices: this.config.gpuIndices,
                rejectActive: !this.config.useActiveGpu,
            },
        };

        const success = await this.trialKeeper.createTrial(opts);
        if (success) {
            this.log.info('Created trial', trialId);
            return trialId;
        } else {
            this.log.warning('Failed to create trial');
            return null;
        }
    }

    public async stopTrial(trialId: string): Promise<void> {
        this.log.info('Stop trial', trialId);
        await this.trialKeeper.stopTrial(trialId);
    }

    public async sendParameter(trialId: string, parameter: Parameter): Promise<void> {
        this.log.info('Trial parameter:', trialId, parameter);
        const command = { type: 'parameter', parameter };
        await this.trialKeeper.sendCommand(trialId, command);
    }

    public onTrialStart(callback: (trialId: string, timestamp: number) => Promise<void>): void {
        this.trialKeeper.onTrialStart(callback);
    }

    public onTrialEnd(callback: (trialId: string, timestamp: number, exitCode: number | null) => Promise<void>): void {
        this.trialKeeper.onTrialStop(callback);
    }

    public onRequestParameter(callback: (trialId: string) => Promise<void>): void {
        this.trialKeeper.onReceiveCommand('request_parameter', (trialId, _command) => {
            callback(trialId);
        });
    }

    public onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void {
        this.trialKeeper.onReceiveCommand('metric', (trialId, command) => {
            callback(trialId, (command as any)['metric']);
        });
    }

    public onEnvironmentUpdate(_callback: (environments: EnvironmentInfo[]) => Promise<void>): void {
        // never
    }

    public async downloadTrialDirectory(trialId: string): Promise<string> {
        // FIXME: hack
        return path.join(globals.paths.experimentRoot, 'environments', this.env.id, 'trials', trialId);
    }
}

// Temporary helpers, will be moved later

import { uniqueString } from 'common/utils';

function uuid(): string {
    return uniqueString(5);
}
