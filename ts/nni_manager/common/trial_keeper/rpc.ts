// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Helper class to manipulate a trial keeper through WebSocket command channel.
 *  Provides similar interface to TrialKeeper class.
 *
 *  Example:
 *
 *      const trialKeeper = new RemoteTrialKeeper(...);
 *      trialKeeper.onTrialStart(...);
 *
 *      server.onConnection((channelId, channel) => {
 *          await trialKeeper.setChannel(channel);
 *          await trialKeeper.start();
 *          await trialKeeper.createTrial(...);
 *      });
 *
 *  Note that "onSomeEvent()" methods should be called before `setChannel()`;
 *  any other methods must be called after `setChannel()`.
 *
 *  Because the remote OS might differ from the local OS,
 *  all path parameters will be changed internally to use '/' as separator.
 **/

import { EventEmitter } from 'node:events';

import type { Command } from 'common/command_channel/interface';
import { RpcHelper, getRpcHelper } from 'common/command_channel/rpc_util';
import type { WsChannel } from 'common/command_channel/websocket/channel';
import { Deferred } from 'common/deferred';
import type { EnvironmentInfo } from 'common/training_service_v3';
import { TrialKeeper } from './keeper';

/**
 *  The trial keeper daemon must call this function before `channel.connect()`.
 **/
export function registerTrialKeeperOnChannel(channel: WsChannel): void {
    getRpcHelper(channel).registerClass('TrialKeeper', TrialKeeper);
}

export class RemoteTrialKeeper {
    private args: any[];
    private emitter: EventEmitter = new EventEmitter();
    private id!: number;
    private initialized: Deferred<void> = new Deferred();
    private rpc!: RpcHelper;

    constructor(environmentId: string, platform: string, enableGpuScheduling: boolean) {
        this.args = [ environmentId, platform, enableGpuScheduling ];
    }

    public async setChannel(channel: WsChannel): Promise<void> {
        this.rpc = getRpcHelper(channel);
        this.id = await this.rpc.construct('TrialKeeper', this.args);

        await Promise.all([
            this.rpc.call(this.id, 'onTrialStart', undefined, [ this.emitTrialStart.bind(this) ]),
            this.rpc.call(this.id, 'onTrialStop', undefined, [ this.emitTrialStop.bind(this) ]),
            this.rpc.call(this.id, 'onReceiveCommand', undefined, [ this.emitCommand.bind(this) ]),
            this.rpc.call(this.id, 'onEnvironmentUpdate', undefined, [ this.emitEnvUpdate.bind(this) ]),
        ]);

        this.initialized.resolve();
    }

    public async start(): Promise<EnvironmentInfo> {
        await this.initialized.promise;
        return await this.rpc.call(this.id, 'start');
    }

    public async shutdown(): Promise<void> {
        await this.rpc.call(this.id, 'shutdown');
    }

    public async registerDirectory(name: string, path: string): Promise<void> {
        await this.rpc.call(this.id, 'registerDirectory', [ name, path.replaceAll('\\', '/') ]);
    }

    public async unpackDirectory(name: string, tarPath: string): Promise<void> {
        await this.rpc.call(this.id, 'unpackDirectory', [ name, tarPath.replaceAll('\\', '/') ]);
    }

    public async createTrial(options: TrialKeeper.TrialOptions): Promise<boolean> {
        return await this.rpc.call(this.id, 'createTrial', [ options ]);
    }

    public async stopTrial(trialId: string): Promise<void> {
        await this.rpc.call(this.id, 'stopTrial', [ trialId ]);
    }

    public async sendCommand(trialId: string, command: Command): Promise<void> {
        await this.rpc.call(this.id, 'sendCommand', [ trialId, command ]);
    }

    public onTrialStart(callback: (trialId: string, timestamp: number) => void): void {
        this.emitter.on('__trial_start', callback);
    }

    private emitTrialStart(trialId: string, timestamp: number): void {
        this.emitter.emit('__trial_start', trialId, timestamp);
    }

    public onTrialStop(callback: (trialId: string, timestamp: number, exitCode: number | null) => void): void {
        this.emitter.on('__trial_stop', callback);
    }

    private emitTrialStop(trialId: string, timestamp: number, exitCode: number | null): void {
        this.emitter.emit('__trial_stop', trialId, timestamp, exitCode);
    }

    public onReceiveCommand(commandType: string, callback: (trialId: string, command: Command) => void): void {
        this.emitter.on(commandType, callback);
    }

    private emitCommand(trialId: string, command: Command): void {
        this.emitter.emit(command.type, trialId, command);
    }

    public onEnvironmentUpdate(callback: (environmentInfo: EnvironmentInfo) => void): void {
        this.emitter.on('__env_update', callback);
    }

    private emitEnvUpdate(envInfo: EnvironmentInfo): void {
        this.emitter.emit('__env_update', envInfo);
    }
}
