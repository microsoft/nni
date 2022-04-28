// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  The IPC channel between NNI manager and tuner.
 *
 *  TODO:
 *   1. Merge with environment service's WebSocket channel.
 *   2. Split import data command to avoid extremely long message.
 *   3. Refactor message format.
 **/

import assert from 'assert/strict';
import { EventEmitter } from 'events';

import { Deferred } from 'ts-deferred';
import type WebSocket from 'ws';

import { Logger, getLogger } from 'common/log';

const logger: Logger = getLogger('tuner_command_channel.WebSocketChannel');

export interface WebSocketChannel {
    init(): Promise<void>;
    shutdown(): Promise<void>;
    sendCommand(command: string): void;  // maybe this should return Promise<void>
    onCommand(callback: (command: string) => void): void;
    onError(callback: (error: Error) => void): void;
}

/**
 *  Get the singleton tuner command channel.
 *  Remember to invoke ``await channel.init()`` before doing anything else.
 **/
export function getWebSocketChannel(): WebSocketChannel {
    return channelSingleton;
}

/**
 *  The callback to serve WebSocket connection request. Used by REST server module.
 *  It should only be invoked once, or an error will be raised.
 *
 *  Type hint of express-ws is somewhat problematic. Don't want to waste time on it so use `any`.
 **/
export function serveWebSocket(ws: WebSocket): void {
    channelSingleton.setWebSocket(ws);
}

class WebSocketChannelImpl implements WebSocketChannel {
    private deferredInit: Deferred<void> = new Deferred<void>();
    private emitter: EventEmitter = new EventEmitter();
    private heartbeatTimer!: NodeJS.Timer;
    private serving: boolean = false;
    private waitingPong: boolean = false;
    private ws!: WebSocket;

    public setWebSocket(ws: WebSocket): void {
        if (this.ws !== undefined) {
            logger.error('A second client is trying to connect');
            ws.close(4030, 'Already serving a tuner.');
            return;
        }

        logger.debug('Connected.');
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
        logger.debug(this.ws === undefined ? 'Waiting connection...' : 'Initialized.');
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

        logger.debug('Sending', command);
        this.ws.send(command);

        if (this.ws.bufferedAmount > command.length + 1000) {
            logger.warning('Sending too fast! Try to reduce the frequency of intermediate results.');
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
        logger.debug('Received', data);
        this.emitter.emit('command', data.toString());
    }

    private handleError(error: Error): void {
        if (!this.serving) {
            logger.debug('Silent error:', error);
            return;
        }
        logger.error('Error:', error);

        clearInterval(this.heartbeatTimer);
        this.emitter.emit('error', error);
        this.serving = false;
    }
}

const channelSingleton: WebSocketChannelImpl = new WebSocketChannelImpl();

let heartbeatInterval: number = 5000;

export namespace UnitTestHelpers {
    export function setHeartbeatInterval(ms: number): void {
        heartbeatInterval = ms;
    }
}
