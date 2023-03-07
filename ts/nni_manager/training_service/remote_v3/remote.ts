// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { setTimeout } from 'node:timers/promises';

import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';
import type { LocalConfig, TrainingServiceConfig } from 'common/experimentConfig';
import type { EnvironmentInfo, Metric, Parameter, TrainingServiceV3 } from 'common/training_service_v3';
import { TrialKeeper } from 'common/trial_keeper/keeper';

import { WsChannelServer } from 'common/command_channel/websocket/server';
import { WsChannelClient } from 'common/command_channel/websocket/client';
import { WsChannel } from 'common/command_channel/websocket/channel';
import { RpcHelper, getRpcHelper, RemoteTrialKeeper } from 'common/command_channel/rpc_util';

class DebugRpc {
    private a: string;

    constructor(a: string) {
        this.a = a;
    }

    public foo(b: string): string {
        return this.a + b;
    }
}

class DebugRpc2 {
    private args: any[];
    private id!: number;
    private rpc: RpcHelper;

    constructor(channel: WsChannel, a: string) {
        this.rpc = getRpcHelper(channel);
        this.rpc.registerClass('DebugRpc', DebugRpc);
        this.args = [ a ];
    }

    public async init(): Promise<void> {
        this.id = await this.rpc.construct('DebugRpc', this.args);
    }

    public async foo(b: string): Promise<string> {
        return await this.rpc.call(this.id, 'foo', [ b ]);
    }
}

export class RemoteTrainingServiceV3 implements TrainingServiceV3 {
    private config: LocalConfig;
    private env: EnvironmentInfo;
    private log: Logger;
    //private trialKeeper: TrialKeeper;
    private trialKeeper!: RemoteTrialKeeper;
    private waitTrialKeeper: Deferred<RemoteTrialKeeper> = new Deferred();

    private server: WsChannelServer;
    private client: WsChannelClient;

    constructor(trainingServiceId: string, config: TrainingServiceConfig) {
        this.log = getLogger(`RemoteV3.${trainingServiceId}`);
        this.log.debug('Training sevice config:', config);

        this.config = config as LocalConfig;
        this.env = { id: `${trainingServiceId}-env` };
        //this.trialKeeper = new TrialKeeper(this.env.id, 'local', Boolean(config.trialGpuNumber));

        this.server = new WsChannelServer('remote_trialkeeper', `platform/${trainingServiceId}`);
        this.server.on('connection', (channelId: string, channel: WsChannel) => {
            this.log.warning('Connection:', channelId);
            console.info('## Connection:', channelId);

            setTimeout(1000).then(() => {
                this.trialKeeper = new RemoteTrialKeeper(channel, this.env.id, 'local', Boolean(this.config.trialGpuNumber));
                this.trialKeeper.init().then(() => {
                    this.waitTrialKeeper.resolve(this.trialKeeper);
                });
            });

            //const obj = new DebugRpc2(channel, 'server-A');
            //obj.init().then(() => {
            //    obj.foo('server-B').then(s => {
            //        this.log.warning('RPC response:', s);
            //        console.log('## RPC response:', s);
            //    });
            //});
        });

        this.client = new WsChannelClient(`ws://localhost:8080/platform/${trainingServiceId}/env`, 'client');
    }

    public async init(): Promise<void> {
        return;
    }

    public async start(): Promise<EnvironmentInfo[]> {
        this.log.info('Start');
        await this.server.start();
        await this.client.connect();

        getRpcHelper(this.client).registerClass('TrialKeeper', TrialKeeper);
        //const obj = new DebugRpc2(this.client, 'client-A');  // register

        await setTimeout(2000);
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
        await this.trialKeeper.registerDirectory(directoryName, path);
    }

    public async createTrial(_envId: string, trialCommand: string, directoryName: string, sequenceId?: number):
            Promise<string | null> {

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
        this.waitTrialKeeper.promise.then(trialKeeper => {
            trialKeeper.onTrialStart(callback);
        });
    }

    public onTrialEnd(callback: (trialId: string, timestamp: number, exitCode: number | null) => Promise<void>): void {
        this.waitTrialKeeper.promise.then(trialKeeper => {
            trialKeeper.onTrialStop(callback);
        });
    }

    public onRequestParameter(callback: (trialId: string) => Promise<void>): void {
        this.waitTrialKeeper.promise.then(trialKeeper => {
            trialKeeper.onReceiveCommand('request_parameter', (trialId, _command) => {
                callback(trialId);
            });
        });
    }

    public onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void {
        this.waitTrialKeeper.promise.then(trialKeeper => {
            trialKeeper.onReceiveCommand('metric', (trialId, command) => {
                callback(trialId, (command as any)['metric']);
            });
        });
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
