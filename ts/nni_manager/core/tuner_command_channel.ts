// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EventEmitter } from 'node:events';

import { WsChannel, WsChannelServer } from 'common/command_channel/websocket';
import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';

export interface IpcInterface {
    init(): Promise<void>;
    sendCommand(commandType: string, content?: string): void;
    onCommand(listener: (commandType: string, content: string) => void): void;
    onError(listener: (error: Error) => void): void;
}

export function getTunerServer(): IpcInterface {
    return server;
}

const logger: Logger = getLogger('tuner_command_channel');

class TunerServer {
    private channel!: WsChannel;
    private connect: Deferred<void> = new Deferred();
    private emitter: EventEmitter = new EventEmitter();
    private server: WsChannelServer;

    constructor() {
        this.server = new WsChannelServer('tuner', 'tuner');
        this.server.onConnection((_channelId, channel) => {
            this.channel = channel;
            this.channel.onError(error => {
                this.emitter.emit('error', error);
            });
            this.channel.onReceive(command => {
                if (command.type === 'ER') {
                    this.emitter.emit('error', new Error(command.content));
                } else {
                    this.emitter.emit('command', command.type, command.content ?? '');
                }
            });
            this.connect.resolve();
        });
        this.server.start();
    }

    public init(): Promise<void> {  // wait connection
        if (this.connect.settled) {
            logger.debug('Initialized.');
            return Promise.resolve();
        } else {
            logger.debug('Waiting connection...');
            // TODO: This is a quick fix. It should check tuner's process status instead.
            setTimeout(() => {
                if (!this.connect.settled) {
                    const msg = 'Tuner did not connect in 10 seconds. Please check tuner (dispatcher) log.';
                    this.connect.reject(new Error('tuner_command_channel: ' + msg));
                }
            }, 10000);
            return this.connect.promise;
        }
    }

    // TODO: for unit test only
    public async stop(): Promise<void> {
        await this.server.shutdown();
    }

    public sendCommand(commandType: string, content?: string): void {
        if (commandType === 'PI') {  // ping is handled with WebSocket protocol
            return;
        }

        if (this.channel.getBufferedAmount() > 1000) {
            logger.warning('Sending too fast! Try to reduce the frequency of intermediate results.');
        }

        this.channel.send({ type: commandType, content });

        if (commandType === 'TE') {
            this.channel.close('TE command');
            this.server.shutdown();
        }
    }

    public onCommand(listener: (commandType: string, content: string) => void): void {
        this.emitter.on('command', listener);
    }

    public onError(listener: (error: Error) => void): void {
        this.emitter.on('error', listener);
    }
}

let server: TunerServer = new TunerServer();

export namespace UnitTestHelpers {
    export function reset(): void {
        server = new TunerServer();
    }

    export async function stop(): Promise<void> {
        await server.stop();
    }
}
