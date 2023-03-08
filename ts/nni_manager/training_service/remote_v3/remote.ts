// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import child_process from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs/promises';
import path from 'node:path';

import { Deferred } from 'common/deferred';
import { globals } from 'common/globals';
import { Logger, getLogger } from 'common/log';
import type { LocalConfig, TrainingServiceConfig } from 'common/experimentConfig';
import type { EnvironmentInfo, Metric, Parameter, TrainingServiceV3 } from 'common/training_service_v3';
import { TrialKeeper } from 'common/trial_keeper/keeper';

import { WsChannelServer } from 'common/command_channel/websocket/server';
import { WsChannelClient } from 'common/command_channel/websocket/client';
import { WsChannel } from 'common/command_channel/websocket/channel';

import { RemoteTrialKeeper, registerForChannel } from 'common/trial_keeper/rpc';

class Worker {
    trainingServiceId: string;
    channelId: string;
    trialKeeper: RemoteTrialKeeper;

    constructor(trainingServiceId: string, channelId: string) {
        this.trainingServiceId = trainingServiceId;
        this.channelId = channelId;
        this.trialKeeper = new RemoteTrialKeeper(this.envId, 'remote', false);  // false <- Boolean(config.trialGpuNumber)
    }

    get envId(): string {
        return `${this.trainingServiceId}-worker${this.channelId}`;
    }

    getEnv(): EnvironmentInfo {
        return { id: this.envId };
    }

    setChannel(channel: WsChannel): void {
        this.trialKeeper.setChannel(channel);
    }

    async start(): Promise<void> {
        const config = {
            experimentId: globals.args.experimentId,
            experimentsDirectory: '/home/lz/nni-experiments',
            logLevel: 'trace',
            pythonInterpreter: '/usr/bin/python',
            platform: 'remote',
            environmentId: this.envId,
            managerCommandChannel: `ws://localhost:8080/platform/${this.trainingServiceId}/${this.channelId}`,
        };
        const configPath = path.join(globals.paths.experimentRoot, 'environments', this.envId, 'trial_keeper_config.json');
        await fs.mkdir(path.dirname(configPath), { recursive: true });
        await fs.writeFile(configPath, JSON.stringify(config));

        const proc = child_process.spawn('python', ['-m', 'nni.tools.nni_manager_scripts.launch_trial_keeper', configPath]);
        // fixme: windows to linux path

        //const client = new WsChannelClient(`ws://localhost:8080/platform/${this.trainingServiceId}/${this.channelId}`);
        //registerForChannel(client);
        //await client.connect();

        await this.trialKeeper.start();
    }

    async stop(): Promise<void> {
        await this.trialKeeper.shutdown();
    }

    async uploadDirectory(name: string, path: string): Promise<void> {
        await this.trialKeeper.registerDirectory(name, path);
    }
}

export class RemoteTrainingServiceV3 implements TrainingServiceV3 {
    private id: string;
    private config: LocalConfig;
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
        this.config = config as LocalConfig;

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
        this.log.info('Start');
        await this.server.start();
        const worker = await this.launchWorker();
        return [ worker.getEnv() ];
    }

    private async launchWorker(): Promise<Worker> {
        this.lastWorkerIndex += 1;
        const worker = new Worker(this.id, String(this.lastWorkerIndex));

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

        await worker.start();
        return worker;
    }

    public async stop(): Promise<void> {
        await Promise.all(this.workers.map(worker => worker.stop()));
        this.log.info('All workers stopped');
    }

    public async uploadDirectory(name: string, path: string): Promise<void> {
        this.log.info(`Upload directory ${name} = ${path}`);
        await Promise.all(this.workers.map(worker => worker.uploadDirectory(name, path)));
    }

    public async createTrial(envId: string, trialCommand: string, directoryName: string, sequenceId?: number):
            Promise<string | null> {

        const worker = this.workersByEnv.get(envId)!;
        const trialId = uuid();

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
        this.emitter.on('trial_end', callback);
    }

    public onRequestParameter(callback: (trialId: string) => Promise<void>): void {
        this.emitter.on('request_parameter', callback);
    }

    public onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void {
        this.emitter.on('metric', callback);
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
