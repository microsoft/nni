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
import { createTarball, getTarballPath } from 'common/tarball';
import type { EnvironmentInfo, Metric, Parameter, TrainingServiceV3 } from 'common/training_service_v3';

import { Worker } from './worker';

export class RemoteTrainingServiceV3 implements TrainingServiceV3 {
    private id: string;
    private config: RemoteConfig;
    private emitter: EventEmitter = new EventEmitter();
    private lastWorkerIndex: number = 0;
    private log: Logger;
    private server: WsChannelServer;
    private uploadedDirs: Set<string> = new Set();
    private workers: Worker[] = [];
    private workersByChannel: Map<string, Worker> = new Map();
    private workersByEnv: Map<string, Worker> = new Map();
    private workersByTrial: Map<string, Worker> = new Map();

    constructor(trainingServiceId: string, config: TrainingServiceConfig) {
        this.id = trainingServiceId;
        this.config = config as RemoteConfig;

        this.log = getLogger(`RemoteV3.${this.id}`);
        this.log.debug('Training sevice config:', config);

        this.server = new WsChannelServer(this.id, `/platform/${this.id}`);

        this.server.on('connection', (channelId: string, channel: WsChannel) => {
            const worker = this.workersByChannel.get(channelId);
            if (worker) {
                worker.setChannel(channel);
                channel.onClose(reason => {
                    this.log.error('Worker channel closed unexpectedly:', reason);
                });
                channel.onError(error => {
                    this.log.error('Worker channel error:', error);
                    this.restartWorker(worker);
                });
            } else {
                this.log.error('Incoming connection from unexpected worker', channelId);
            }
        });
    }

    public async init(): Promise<void> {
        return;
    }

    public async start(): Promise<EnvironmentInfo[]> {
        this.log.info('Starting remote training service...');
        await this.server.start();
        await Promise.all(
            this.config.machineList.map(workerConfig => this.launchWorker(workerConfig))
        );
        this.log.info('Remote training service started');
        return this.workers.map(worker => worker.env);
    }

    public async stop(): Promise<void> {
        await Promise.allSettled(this.workers.map(worker => worker.stop()));
        this.log.info('All workers stopped');
    }

    public async uploadDirectory(name: string, path: string): Promise<void> {
        this.log.info(`Upload directory ${name} = ${path}`);
        const tar = await createTarball(name, path);  // TODO: this should be done outside
        this.uploadedDirs.add(name);

        // since upload() is async, when uploaded this.workers might have already changed
        const workers = Array.from(this.workers);

        const results = await Promise.allSettled(
            workers.map(worker => worker.upload(name, tar))
        );

        // TODO: currently this is only called on start up, so it's acceptable to skip recovering
        let fail = false;
        results.forEach((result, i) => {
            if (result.status === 'rejected') {
                this.log.error(`Worker ${workers[i].envId} failed to upload ${name}:`, result.reason);
                this.stopWorker(workers[i], false);
                fail = true;
            }
        });
        if (fail) {
            this.emitEnvUpdate();
        }
    }

    public async createTrial(
        envId: string,
        trialCommand: string,
        directoryName: string,
        sequenceId: number,
        trialId?: string
    ): Promise<string | null> {
        const worker = this.workersByEnv.get(envId);
        if (!worker) {
            this.log.warning('Cannot create trial. Bad environment ID:', envId);
            return null;
        }

        trialId = trialId ?? uuid();

        let gpuNumber = this.config.trialGpuNumber;
        if (gpuNumber) {
            gpuNumber /= worker.config.maxTrialNumberPerGpu;
        }

        const opts: TrialKeeper.TrialOptions = {
            id: trialId,
            command: trialCommand,
            codeDirectoryName: directoryName,
            sequenceId,  // TODO: move to global counter
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
        const worker = this.workersByTrial.get(trialId);
        await worker?.trialKeeper.stopTrial(trialId);
    }

    public async sendParameter(trialId: string, parameter: Parameter): Promise<void> {
        this.log.info('Trial parameter:', trialId, parameter);
        const worker = this.workersByTrial.get(trialId);
        if (!worker) {
            this.log.error(`Worker of trial ${trialId} is not working`);
            return;
        }
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

    private emitEnvUpdate(): void {
        this.emitter.emit('env_update', this.workers.map(worker => worker.env));
    }

    private async launchWorker(config: RemoteMachineConfig): Promise<Worker> {
        this.lastWorkerIndex += 1;  // TODO: add a global counter module to support recover on resume
        const channelId = String(this.lastWorkerIndex);
        const worker = new Worker(
            this.id,
            channelId,
            config,
            this.server.getChannelUrl(channelId, this.config.nniManagerIp),
            Boolean(this.config.trialGpuNumber)
        );

        this.workers.push(worker);
        this.workersByChannel.set(worker.channelId, worker);
        this.workersByEnv.set(worker.envId, worker);

        worker.trialKeeper.onTrialStart((trialId, timestamp) => {
            this.emitter.emit('trial_start', trialId, timestamp);
        });
        worker.trialKeeper.onTrialStop((trialId, timestamp, exitCode) => {
            //if (this.shouldCollectLog(exitCode !== 0)) {
            //    this.collectTrialLog(trialId);
            //}
            this.emitter.emit('trial_stop', trialId, timestamp, exitCode);
            // do not delete workersByTrial. it might be used to collect logs
        });
        worker.trialKeeper.onReceiveCommand('request_parameter', (trialId, _command) => {
            this.emitter.emit('request_parameter', trialId);
        });
        worker.trialKeeper.onReceiveCommand('metric', (trialId, command) => {
            this.emitter.emit('metric', trialId, command['metric']);
        });
        worker.trialKeeper.onEnvironmentUpdate(env => {
            worker.env = env;
            this.emitEnvUpdate();
        });

        await worker.start();
        return worker;
    }

    private stopWorker(oldWorker: Worker, emitEnvUpdate: boolean): void {
        this.workers = this.workers.filter(worker => (worker !== oldWorker));
        if (emitEnvUpdate) {
            this.emitEnvUpdate();
        }

        this.workersByChannel.delete(oldWorker.channelId);
        this.workersByEnv.delete(oldWorker.envId);

        const now = Date.now();
        this.workersByTrial.forEach((worker, trialId) => {
            if (worker === oldWorker) {
                this.emitter.emit('trial_stop', trialId, now, null);
                this.workersByTrial.delete(trialId);  // mdn says it's save
            }
        });
    }

    private async restartWorker(oldWorker: Worker): Promise<void> {
        this.stopWorker(oldWorker, true);

        try {
            const worker = await this.launchWorker(oldWorker.config);

            for (const dirName of this.uploadedDirs) {
                const tar = getTarballPath(dirName);
                await worker.upload(dirName, tar);
            }
            
        } catch (error) {
            this.log.error(`Failed to recover worker ${oldWorker.config.host}:`, error);
            return;
        }

        this.emitEnvUpdate();
        this.log.info(`Worker ${oldWorker.config.host} has been recovered`);
    }

    /*  used to debug pipeline. re-enable it when we support log collection

    private shouldCollectLog(errorOccurred: boolean): boolean {
        if (this.config.logCollection === 'always') {
            return true;
        }
        if (this.config.logCollection === 'never') {
            return false;
        }
        return errorOccurred;
    }
    */

    public async downloadTrialDirectory(trialId: string): Promise<string> {
        const worker = this.workersByTrial.get(trialId);
        if (worker) {
            return await worker.downloadTrialLog(trialId);  // TODO: should download NNI_OUTPUT_DIR as well
        } else {
            this.log.error('Failed to download trial log: cannot find worker for trial', trialId);
            throw new Error(`The worker of trial ${trialId} is not working`);
        }
    }
}

// Temporary helpers, will be moved later

import { uniqueString } from 'common/utils';

function uuid(): string {
    return uniqueString(5);
}
