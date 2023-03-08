// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import child_process from 'node:child_process';
import { EventEmitter, once } from 'node:events';
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

import { createTarball } from 'common/tarball';

import { Ssh } from './ssh';

class Worker {
    trainingServiceId: string;
    channelId: string;
    channelUrl: string;
    trialKeeper: RemoteTrialKeeper;
    ssh: Ssh;
    launchResult!: any;

    public get envId(): string {
        return `${this.trainingServiceId}-worker${this.channelId}`;
    }

    constructor(trainingServiceId: string, channelId: string, channelUrl: string) {
        this.trainingServiceId = trainingServiceId;
        this.channelId = channelId;
        this.channelUrl = channelUrl;
        this.trialKeeper = new RemoteTrialKeeper(this.envId, 'remote', false);  // false <- Boolean(config.trialGpuNumber)
        this.ssh = new Ssh({
            host: 'localhost',
            password: 'cffbk',
            port: 22,
            //privateKey: key,
            username: 'lz',
        });
    }

    getEnv(): EnvironmentInfo {
        return { id: this.envId };
    }

    public setChannel(channel: WsChannel): void {
        this.trialKeeper.setChannel(channel);
    }

    public async start(): Promise<void> {
        const config = {
            experimentId: globals.args.experimentId,
            logLevel: globals.args.logLevel,
            platform: 'remote',
            environmentId: this.envId,
            managerCommandChannel: this.channelUrl,
        };

        const initDir = '/tmp/nni-debug2'
        await fs.mkdir(initDir, { recursive: true });
        await fs.writeFile(path.join(initDir, 'config.json'), JSON.stringify(config));

        const cmd = `python -m nni.tools.nni_manager_scripts.launch_trial_keeper ${initDir}`;

        const deferred = new Deferred<void>();

        await this.ssh.connect();
        const result = await this.ssh.exec(cmd);

        await this.ssh.download(path.join(initDir, 'launch.json'), '/tmp/launch.json');

        const output = await fs.readFile('/tmp/launch.json', { encoding: 'utf8' });
        console.log('## launch trial keeper result:', output);
        this.launchResult = JSON.parse(output);

        await this.trialKeeper.start();
    }

    async stop(): Promise<void> {
        await this.trialKeeper.shutdown();
    }

    async upload(name: string, tar: string): Promise<void> {
        const remotePath = path.join(this.launchResult.envDir, 'upload', `${name}.tgz`);
        await this.ssh.upload(tar, remotePath);
        await this.trialKeeper.unpackDirectory(name, remotePath);
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
        const channelId = String(this.lastWorkerIndex);
        const worker = new Worker(this.id, channelId, this.server.getChannelUrl(channelId));

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
        const tar = await createTarball(name, path);
        await Promise.all(this.workers.map(worker => worker.upload(name, tar)));
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
        this.emitter.on('trial_stop', callback);
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
