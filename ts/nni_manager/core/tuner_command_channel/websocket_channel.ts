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

import type WebSocket from 'ws';

import { Deferred } from 'common/deferred';
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
 *  If it is invoked more than once, the previous connection will be dropped.
 **/
export function serveWebSocket(ws: WebSocket): void {
    channelSingleton.serveWebSocket(ws);
}

class WebSocketChannelImpl implements WebSocketChannel {
    private deferredInit: Deferred<void> = new Deferred<void>();
    private emitter: EventEmitter = new EventEmitter();
    private heartbeatTimer!: NodeJS.Timer;
    private serving: boolean = false;
    private waitingPong: boolean = false;
    private ws!: WebSocket;

    public serveWebSocket(ws: WebSocket): void {
        if (this.ws === undefined) {
            logger.debug('Connected.');
        } else {
            logger.warning('Reconnecting. Drop previous connection.');
            this.dropConnection('Reconnected');
        }

        this.serving = true;

        this.ws = ws;
        this.ws.on('close', this.handleWsClose);
        this.ws.on('error', this.handleWsError);
        this.ws.on('message', this.handleWsMessage);
        this.ws.on('pong', this.handleWsPong);

        this.heartbeatTimer = setInterval(this.heartbeat.bind(this), heartbeatInterval);
        this.deferredInit.resolve();
    }

    public init(): Promise<void> {
        if (this.ws === undefined) {
            logger.debug('Waiting connection...');
            // TODO: This is a quick fix. It should check tuner's process status instead.
            setTimeout(() => {
                if (!this.deferredInit.settled) {
                    const msg = 'Tuner did not connect in 10 seconds. Please check tuner (dispatcher) log.';
                    this.deferredInit.reject(new Error('tuner_command_channel: ' + msg));
                }
            }, 10000);
            return this.deferredInit.promise;

        } else {
            logger.debug('Initialized.');
            return Promise.resolve();
        }
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

    /* Following callbacks must be auto-binded arrow functions to be turned off */

    private handleWsClose = (): void => {
        this.handleError(new Error('tuner_command_channel: Tuner closed connection'));
    }

    private handleWsError = (error: Error): void => {
        this.handleError(error);
    }

    private handleWsMessage = (data: Buffer, _isBinary: boolean): void => {
        this.receive(data);
    }

    private handleWsPong = (): void => {
        this.waitingPong = false;
    }

    private dropConnection(reason: string): void {
        if (this.ws === undefined) {
            return;
        }

        this.serving = false;
        this.waitingPong = false;
        clearInterval(this.heartbeatTimer);

        this.ws.off('close', this.handleWsClose);
        this.ws.off('error', this.handleWsError);
        this.ws.off('message', this.handleWsMessage);
        this.ws.off('pong', this.handleWsPong);

        this.ws.on('close', () => {
            logger.info('Connection dropped');
        });
        this.ws.on('message', (data, _isBinary) => {
            logger.error('Received message after reconnect:', data);
        });
        this.ws.on('pong', () => {
            logger.error('Received pong after reconnect.');
        });
        this.ws.close(1001, reason);
    }

    private heartbeat(): void {
        if (this.waitingPong) {
            this.ws.terminate();  // this will trigger "close" event
            this.handleError(new Error('tuner_command_channel: Tuner loses responsive'));
        }

        this.waitingPong = true;
        this.ws.ping();
    }

    private receive(data: Buffer): void {
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
