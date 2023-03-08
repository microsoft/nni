// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import child_process from 'node:child_process';
import { EventEmitter, once } from 'node:events';
import fs from 'node:fs/promises';
import path from 'node:path';

import tar from 'tar';

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

import { Client, ClientChannel, SFTPWrapper } from 'ssh2';

class Worker {
    trainingServiceId: string;
    channelId: string;
    channelUrl: string;
    trialKeeper: RemoteTrialKeeper;
    ssh: Client;
    launchResult!: any;

    constructor(trainingServiceId: string, channelId: string, channelUrl: string) {
        this.trainingServiceId = trainingServiceId;
        this.channelId = channelId;
        this.channelUrl = channelUrl;
        this.trialKeeper = new RemoteTrialKeeper(this.envId, 'remote', false);  // false <- Boolean(config.trialGpuNumber)
        this.ssh = new Client();
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

        //const key = await fs.readFile('/home/lz/.ssh/id_ed25519', { encoding: 'utf8' });

        this.ssh.connect({
            host: 'localhost',
            password: 'cffbk',
            port: 22,
            //privateKey: key,
            username: 'lz',
        });
        await once(this.ssh, 'ready');
        this.ssh.exec(cmd, (error, stream) => {
            if (error) {
                deferred.reject(error);
            } else {
                stream.on('data', (data: any) => { console.log('SSH stdout:', String(data)); });
                stream.stderr.on('data', (data: any) => { console.log('SSH stderr:', String(data)); });
                stream.on('close', (code: any, signal: any) => {
                    console.log('SSH result:', code, signal);
                    deferred.resolve();
                });
            }
        });

        await deferred.promise;

        const output = await fs.readFile(path.join(initDir, 'launch.json'), { encoding: 'utf8' });
        console.log('## launch trial keeper result:', output);
        this.launchResult = JSON.parse(output);

        await this.trialKeeper.start();
    }

    async stop(): Promise<void> {
        await this.trialKeeper.shutdown();
    }

    async upload(name: string, localTarPath: string): Promise<void> {
        const deferred = new Deferred<SFTPWrapper>();
        this.ssh.sftp((error, sftp) => {
            if (error) {
                deferred.reject(error);
            } else {
                deferred.resolve(sftp);
            }
        });
        const sftp = await deferred.promise;

        const p = path.join(this.launchResult.envDir, 'upload', `${name}.tgz`);
        const remoteTarPath = p.replaceAll('\\', '/');

        const deferred2 = new Deferred<void>();
        console.log('## scp', localTarPath, remoteTarPath);
        sftp.fastPut(localTarPath, remoteTarPath, (error: any) => {
            if (error) {
                deferred2.reject(error);
            } else {
                deferred2.resolve();
            }
        });
        await deferred2.promise;

        await this.trialKeeper.unpackDirectory(name, remoteTarPath);
    }
}

async function createTarball(tarName: string, dir: string): Promise<string> {
    const tarDir = path.join(globals.paths.experimentRoot, 'upload');
    await fs.mkdir(tarDir, { recursive: true });
    const tarPath = path.join(tarDir, `${tarName}.tgz`);

    const tarOpts = {
        file: tarPath,
        cwd: dir,
        gzip: true,
        portable: true,
    } as const;
    await tar.create(tarOpts, [ '.' ]);

    return tarPath;
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
        const tarPath = await createTarball(name, path);
        await Promise.all(this.workers.map(worker => worker.upload(name, tarPath)));
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
