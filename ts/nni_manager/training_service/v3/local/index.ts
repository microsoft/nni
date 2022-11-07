// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EventEmitter } from 'events';
import fsPromises from 'fs/promises';
import path from 'path';

import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';
import type { LocalConfig, TrainingServiceConfig } from 'common/experimentConfig';
import type { EnvironmentInfo, Metric, Parameter, TrainingServiceV3 } from 'common/training_service_v3';
import { Trial } from './trial';

import { Command, HttpChannelManager } from 'training_service/v3/staging';  // FIXME

type TrialStatus = 'create' | 'running' | 'end' | 'error';

export class LocalTrainingServiceV3 implements TrainingServiceV3 {
    private codeDirs: Map<string, string> = new Map();
    private config: LocalConfig;
    private emitter: EventEmitter = new EventEmitter();
    private env: EnvironmentInfo;
    private logger: Logger;
    private serviceId: string;
    private trials: Map<string, Trial> = new Map();
    private channels!: HttpChannelManager;

    constructor(trainingServiceId: string, config: TrainingServiceConfig) {
        this.serviceId = trainingServiceId;
        this.config = config as LocalConfig;
        this.env = { id: `${trainingServiceId}-env` };
        this.logger = getLogger('LocalV3.' + trainingServiceId);
        this.logger.debug('Training sevice config:', config);
    }

    public async init(): Promise<void> {
        this.channels = new HttpChannelManager(this.env.id);
        this.channels.onReceive((trialId, command) => {
            if (command.type === 'request_parameter') {
                this.emitter.emit('request_parameter', trialId);
            } else if (command.type === 'metric') {
                this.emitter.emit('metric', trialId, command.metric);
            } else {
                this.logger.error('Received bad command:', trialId, command);
            }
        });
    }

    public async start(): Promise<EnvironmentInfo[]> {
        this.logger.info('Start.');
        return [ this.env ];
    }

    public async stop(): Promise<void> {
        const trials = Array.from(this.trials.values());
        const runningTrials = trials.filter(trial => trial.isRunning());
        const promises = trials.map(trial => trial.kill());
        await Promise.all(promises);
        this.logger.info('All trials stopped.');
    }

    public async uploadDirectory(directoryName: string, path: string): Promise<void> {
        this.logger.info(`Register directory ${directoryName} = ${path}`);
        this.codeDirs.set(directoryName, path);
    }

    public async createTrial(_envId: string, trialCommand: string, directoryName: string): Promise<string | null> {
        const trialId = uuid();
        const codeDir = this.codeDirs.get(directoryName);

        this.logger.info('Create trial', trialId);
        this.logger.debug('    command:', trialCommand);
        this.logger.debug('    directory:', directoryName, codeDir);

        // TODO: globals.paths.getEnvironmentTrialDirectory(envId, trialId)
        const outputDir = path.join(globals.paths.experimentRoot, 'environments', this.env.id, 'trials', trialId);
        await fsPromises.mkdir(outputDir, { recursive: true });

        const trial = new Trial(trialId, this.emitter);

        const extra = {
            NNI_PLATFORM: this.config.platform,
            NNI_TRIAL_COMMAND_CHANNEL: buildUrl('http', `/env/${this.env.id}/${trialId}`),
        };

        const success = await trial.spawn(trialCommand, codeDir!, outputDir, extra);

        if (success) {
            this.logger.debug('Created trial', trialId);
            this.trials.set(trialId, trial);
            return trialId;
        } else {
            this.logger.error('Failed to create trail.');
            return null;
        }
    }

    public async stopTrial(trialId: string): Promise<void> {
        this.logger.info('Stop trial', trialId);
        await this.trials.get(trialId)!.kill();
    }

    public async sendParameter(trialId: string, parameter: Parameter): Promise<void> {
        this.logger.info('Trial parameter:', trialId, parameter);
        const command = { type: 'parameter', parameter };
        this.channels.send(trialId, command);
    }

    public onRequestParameter(callback: (trialId: string) => Promise<void>): void {
        this.emitter.on('request_parameter', callback);
    }

    public onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void {
        this.emitter.on('metric', callback);
    }

    public onTrialStart(callback: (trialId: string, timestamp: number) => Promise<void>): void {
        this.emitter.on('trial_start', callback);
    }

    public onTrialEnd(callback: (trialId: string, timestamp: number, exitCode: number | null) => Promise<void>): void {
        this.emitter.on('trial_end', callback);
    }

    public onEnvironmentUpdate(_callback: (environments: EnvironmentInfo[]) => Promise<void>): void {
        // never
    }
}

// Temporary helpers, will be moved later

import { uniqueString } from 'common/utils';

function uuid(): string {
    return uniqueString(5);
}

function buildUrl(protocol: string, suffix: string) {
    if (suffix.startsWith('/')) {
        suffix = suffix.slice(1);
    }
    if (globals.args.urlPrefix) {
        return `${protocol}://localhost:${globals.args.port}/${globals.args.urlPrefix}/${suffix}`;
    } else {
        return `${protocol}://localhost:${globals.args.port}/${suffix}`;
    }
}
