// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Remote training service that runs trials on one or more SSH servers.
 *
 *  It supports both POSIX and Windows servers,
 *  but requires the server to have a recent python installed.
 **/

import { EventEmitter } from 'node:events';

import { WsChannel, WsChannelServer } from 'common/command_channel/websocket/index';
import type { RemoteConfig, RemoteMachineConfig, TrainingServiceConfig } from 'common/experimentConfig';
import type { TrialKeeper } from 'common/trial_keeper/keeper';
import { Logger, getLogger } from 'common/log';
import { createTarball } from 'common/tarball';
import type { EnvironmentInfo, Metric, Parameter, TrainingServiceV3 } from 'common/training_service_v3';

import { Worker } from './worker';

export class RemoteTrainingServiceV3 implements TrainingServiceV3 {
    private id: string;
    private config: RemoteConfig;
    private log: Logger;

    private emitter: EventEmitter = new EventEmitter();

    private lastWorkerIndex: number = 0;
    private workers: Worker[] = [];
    private workersByChannel: Map<string, Worker> = new Map();
    private workersByEnv: Map<string, Worker> = new Map();
    private workersByTrial: Map<string, Worker> = new Map();

    private server: WsChannelServer;

    constructor(trainingServiceId: string, config: TrainingServiceConfig) {
        this.id = trainingServiceId;
        this.config = config as RemoteConfig;

        this.log = getLogger(`RemoteV3.${this.id}`);
        this.log.debug('Training sevice config:', config);

        this.server = new WsChannelServer('remote_trialkeeper', `platform/${this.id}`);

        this.server.on('connection', (channelId: string, channel: WsChannel) => {
            const worker = this.workersByChannel.get(channelId);
            if (worker) {
                worker.setChannel(channel);
            } else {
                this.log.error('Incoming connection from unexpected worker', channelId);
            }
        });
    }

    public async init(): Promise<void> {
        return;
    }

    public async start(): Promise<EnvironmentInfo[]> {
        this.log.info('Starting remote training service');
        await this.server.start();
        for (const workerConfig of this.config.machineList) {
            const worker = await this.launchWorker(workerConfig);
        }
        this.log.info('Remote training service started');
        return this.workers.map(worker => worker.env);
    }

    private async launchWorker(config: RemoteMachineConfig): Promise<Worker> {
        this.lastWorkerIndex += 1;
        const channelId = String(this.lastWorkerIndex);
        const worker = new Worker(
            this.id,
            channelId,
            config,
            this.server.getChannelUrl(channelId),
            Boolean(this.config.trialGpuNumber)
        );

        this.workers.push(worker);
        this.workersByChannel.set(worker.channelId, worker);
        this.workersByEnv.set(worker.envId, worker);

        worker.trialKeeper.onTrialStart((...args) => {
            this.emitter.emit('trial_start', ...args);
        });
        worker.trialKeeper.onTrialStop((...args) => {
            this.emitter.emit('trial_stop', ...args);
        });
        worker.trialKeeper.onReceiveCommand('request_parameter', (trialId, _command) => {
            this.emitter.emit('request_parameter', trialId);
        });
        worker.trialKeeper.onReceiveCommand('metric', (trialId, command) => {
            this.emitter.emit('metric', trialId, command['metric']);
        });
        worker.trialKeeper.onEnvironmentUpdate(env => {
            worker.env = env;
            this.emitter.emit('env_update', this.workers.map(worker => worker.env));
        });

        await worker.start();
        return worker;
    }

    public async stop(): Promise<void> {
        await Promise.all(this.workers.map(worker => worker.stop()));
        this.log.info('All workers stopped');
    }

    public async uploadDirectory(name: string, path: string): Promise<void> {
        this.log.info(`Upload directory ${name} = ${path}`);
        const tar = await createTarball(name, path);
        await Promise.all(this.workers.map(worker => worker.upload(name, tar)));
    }

    public async createTrial(envId: string, trialCommand: string, directoryName: string, sequenceId?: number):
            Promise<string | null> {

        const worker = this.workersByEnv.get(envId)!;
        const trialId = uuid();

        let gpuNumber = this.config.trialGpuNumber;
        if (gpuNumber) {
            gpuNumber /= worker.config.maxTrialNumberPerGpu;
        }

        const opts: TrialKeeper.TrialOptions = {
            id: trialId,
            command: trialCommand,
            codeDirectoryName: directoryName,
            sequenceId,
            gpuNumber,
            gpuRestrictions: {
                onlyUseIndices: worker.config.gpuIndices,
                rejectActive: !worker.config.useActiveGpu,
            },
        };

        const success = await worker.trialKeeper.createTrial(opts);
        if (success) {
            this.log.info(`Created trial ${trialId} on worker ${worker.channelId}`);
            this.workersByTrial.set(trialId, worker);
            return trialId;
        } else {
            this.log.warning('Failed to create trial');
            return null;
        }
    }

    public async stopTrial(trialId: string): Promise<void> {
        this.log.info('Stop trial', trialId);
        const worker = this.workersByTrial.get(trialId)!;
        await worker.trialKeeper.stopTrial(trialId);
    }

    public async sendParameter(trialId: string, parameter: Parameter): Promise<void> {
        this.log.info('Trial parameter:', trialId, parameter);
        const worker = this.workersByTrial.get(trialId)!;
        const command = { type: 'parameter', parameter };
        await worker.trialKeeper.sendCommand(trialId, command);
    }

    public onTrialStart(callback: (trialId: string, timestamp: number) => Promise<void>): void {
        this.emitter.on('trial_start', callback);
    }

    public onTrialEnd(callback: (trialId: string, timestamp: number, exitCode: number | null) => Promise<void>): void {
        this.emitter.on('trial_stop', callback);
    }

    public onRequestParameter(callback: (trialId: string) => Promise<void>): void {
        this.emitter.on('request_parameter', callback);
    }

    public onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void {
        this.emitter.on('metric', callback);
    }

    public onEnvironmentUpdate(callback: (environments: EnvironmentInfo[]) => Promise<void>): void {
        this.emitter.on('env_update', callback);
    }
}

// Temporary helpers, will be moved later

import { uniqueString } from 'common/utils';

function uuid(): string {
    return uniqueString(5);
}
