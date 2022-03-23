// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import { EventEmitter } from 'events';

import { Deferred } from 'ts-deferred';
import type WebSocket from 'ws';

import { Logger, getLogger } from 'common/log';

export function getWebSocketChannel(): WebSocketChannel {
    return channelSingleton;
}

export function serveWebSocket(ws: any, _req: any, _next: any): void {
    // typed of express-ws is somewhat problematic, don't want to waste time on it
    channelSingleton.setWebSocket(ws);
}

// TODO: this class should not be fully exported (export an interface instead)
export class WebSocketChannel {
    private deferredInit: Deferred<void> = new Deferred<void>();
    private emitter: EventEmitter = new EventEmitter();
    private heartbeatTimer!: NodeJS.Timer;
    private logger: Logger = getLogger('tuner_command_channel.WebSocketChannel');
    private serving: boolean = false;
    private waitingPong: boolean = false;
    private ws!: WebSocket;

    public setWebSocket(ws: WebSocket): void {
        if (this.ws !== undefined) {
            this.logger.error('A second client is trying to connect');
            ws.close(4030, 'Already serving a tuner.');
            return;
        }

        this.logger.debug('Connected.');
        this.serving = true;

        this.ws = ws;
        ws.on('close', () => { this.handleError(new Error('tuner_command_channel: Tuner closed connection')); });
        ws.on('error', this.handleError.bind(this));
        ws.on('message', this.receive.bind(this));
        ws.on('pong', () => { this.waitingPong = false; });

        this.heartbeatTimer = setInterval(this.heartbeat.bind(this), heartbeatInterval);
        this.deferredInit.resolve();
    }

    public init(): Promise<void> {
        this.logger.debug(this.ws === undefined ? 'Waiting connection...' : 'Initialized.');
        return this.deferredInit.promise;
    }

    public async shutdown(): Promise<void> {
        if (this.ws === undefined) {
            return;
        }
        clearInterval(this.heartbeatTimer);
        this.serving = false;
        this.emitter.removeAllListeners();
    }

    public sendCommand(command: string): void {
        assert.ok(this.ws !== undefined);

        this.logger.debug('Sending', command);
        this.ws.send(command);

        if (this.ws.bufferedAmount > command.length + 1000) {
            this.logger.warning('Sending too fast! Try to reduce the frequency of intermediate results.');
        }
    }

    public onCommand(callback: (command: string) => void): void {
        this.emitter.on('command', callback);
    }

    public onError(callback: (error: Error) => void): void {
        this.emitter.on('error', callback);
    }

    private heartbeat(): void {
        if (this.waitingPong) {
            this.ws.terminate();  // this will trigger "close" event
            this.handleError(new Error('tuner_command_channel: Tuner loses responsive'));
        }

        this.waitingPong = true;
        this.ws.ping();
    }

    private receive(data: Buffer, _isBinary: boolean): void {
        this.logger.debug('Received', data);
        this.emitter.emit('command', data.toString());
    }

    private handleError(error: Error): void {
        if (!this.serving) {
            this.logger.debug('Silent error:', error);
            return;
        }
        this.logger.error('Error:', error);

        clearInterval(this.heartbeatTimer);
        this.emitter.emit('error', error);
        this.serving = false;
    }
}

const channelSingleton: WebSocketChannel = new WebSocketChannel();

const heartbeatInterval: number = 5000;
