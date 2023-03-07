// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EventEmitter } from 'node:events';
import util from 'node:util';

import type { WebSocket } from 'ws';

import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';

import type { Command } from '../interface';

interface WsChannelEvents {
    'command': (command: Command) => void;
    'close': (reason: string) => void;
    'lost': () => void;
    'error': (error: Error) => void;  // not used in base class
}

export declare interface WsChannel {
    on<E extends keyof WsChannelEvents>(event: E, listener: WsChannelEvents[E]): this;
}

export class WsChannel extends EventEmitter {
    private closed: Deferred<void> | null = null;
    private commandEmitter: EventEmitter = new EventEmitter();
    private connection: WsConnection | null = null;
    private epoch: number = 0;
    private heartbeatInterval: number | null;
    private log: Logger;
    private name: string;

    constructor(name: string, ws?: WebSocket, heartbeatInterval?: number) {
        super()
        this.log = getLogger(`WsChannel.${name}`);
        this.name = name;
        this.heartbeatInterval = heartbeatInterval || null;
        if (ws) {
            this.setConnection(ws);
        }
    }

    public setConnection(ws: WebSocket): void {
        if (this.connection) {
            this.epoch += 1;
        }
        this.connection = this.configConnection(ws);
    }

    public close(reason: string): Promise<void> {
        this.closed = new Deferred<void>();
        if (this.connection) {
            this.connection.close(reason);
        } else {
            this.closed.resolve();
        }
        return this.closed.promise;
    }

    public send(command: Command): void {
        if (this.connection) {
            this.connection.send(command);
        } else {
            // TODO: add a queue?
            this.log.warning('Connection lost; drop command', command);
        }
    }

    public sendAsync(command: Command): Promise<void> {
        if (this.connection) {
            return this.connection.sendAsync(command);
        } else {
            this.log.warning('Connection lost; drop command async', command);
            return Promise.resolve();
        }
    }

    public onCommand(commandType: string, callback: (command: Command) => void): void {
        this.log.debug('## register command callback', commandType);
        this.commandEmitter.on(commandType, callback);
        console.log('## reg:', commandType, this.commandEmitter.listenerCount(commandType));
    }

    private configConnection(ws: WebSocket): WsConnection {
        const epoch = this.epoch;  // copy it to use in closure
        const conn = new WsConnection(
            this.epoch ? `${this.name}.${epoch}` : this.name,
            ws,
            this.commandEmitter,
            this.heartbeatInterval
        );

        conn.on('command', command => {
            this.emit('command', command);
        });
        conn.on('bye', reason => {
            this.connection = null;
            this.epoch += 1;
            if (!this.closed) {
                this.closed = new Deferred<void>();
            }
            this.closed.resolve();
            this.emit('close', reason);
        });
        conn.on('close', (code, reason) => {
            this.closeConnection(epoch, `Received closing handshake: ${code} ${reason}`);
        });
        conn.on('error', error => {
            this.closeConnection(epoch, `Error occurred: ${util.inspect(error)}`);
        });

        return conn;
    }

    private closeConnection(epoch: number, reason: string): void {
        if (this.epoch !== epoch) {
            this.log.debug(`Previous connection closed ${epoch}: ${reason}`);
            return;
        }

        if (this.closed) {
            this.log.debug('Connection closed:', reason);
            this.closed.resolve();
        } else {
            this.log.warning('Connection closed unexpectedly:', reason);
            this.emit('lost');
        }

        this.connection = null;
        this.epoch += 1;
    }
}

interface WsConnectionEvents {
    'command': (command: Command) => void;
    'bye': (reason: string) => void;
    'close': (code: number, reason: string) => void;
    'error': (error: Error) => void;
}

declare interface WsConnection {
    on<E extends keyof WsConnectionEvents>(event: E, listener: WsConnectionEvents[E]): this;
}

class WsConnection extends EventEmitter {
    private closing: boolean = false;
    private commandEmitter: EventEmitter;
    private heartbeatTimer: NodeJS.Timer | null = null;
    private log: Logger;
    private missingPongs: number = 0;
    private ws: WebSocket;

    constructor(name: string, ws: WebSocket, commandEmitter: EventEmitter, heartbeatInterval: number | null) {
        super();
        this.log = getLogger(`WsConnection.${name}`);
        this.ws = ws;
        this.commandEmitter = commandEmitter;

        ws.on('close', this.handleClose.bind(this));
        ws.on('error', this.handleError.bind(this));
        ws.on('message', this.handleMessage.bind(this));
        ws.on('pong', this.handlePong.bind(this));

        if (heartbeatInterval) {
            this.heartbeatTimer = setInterval(this.heartbeat.bind(this), heartbeatInterval);
        }
    }

    public close(reason: string): void {
        if (this.closing) {
            this.log.debug('Closing again:', reason);
        } else {
            this.log.debug('Closing:', reason);
            this.closing = true;
            if (this.heartbeatTimer) {
                clearInterval(this.heartbeatTimer);
                this.heartbeatTimer = null;
            }
            this.sendAsync({ type: '_bye_', reason }).finally(() => {
                this.ws.close(1001, reason);
            });
        }
    }

    public send(command: Command): void {
        this.log.trace('Sending command', command);
        this.ws.send(JSON.stringify(command));
    }

    public sendAsync(command: Command): Promise<void> {
        this.log.trace('Sending command async', command);
        const deferred = new Deferred<void>();
        this.ws.send(JSON.stringify(command), error => {
            if (error) {
                deferred.reject(error);
            } else {
                deferred.resolve();
            }
        });
        return deferred.promise;
    }

    private handleClose(code: number, reason: Buffer): void {
        if (this.closing) {
            this.log.debug('Connection closed');
        } else {
            this.log.debug('Connection closed by peer:', code, String(reason));
            this.emit('close', code, String(reason));
        }
    }
    
    private handleError(error: Error): void {
        if (this.closing) {
            this.log.warning('Error after closing:', error);
        } else {
            this.emit('error', error);
        }
    }

    private handleMessage(data: Buffer, _isBinary: boolean): void {
        const s = String(data);
        if (this.closing) {
            this.log.warning('Received message after closing:', s);
            return;
        }

        this.log.trace('Received command', s);
        const command = JSON.parse(s);

        if (command.type === '_nop_') {
            return;
        }
        if (command.type === '_bye_') {
            this.closing = true;
            this.emit('bye', command.reason);
            return;
        }

        const hasEventListener = this.emit('command', command);
        const hasCommandListener = this.commandEmitter.emit(command.type, command);
        console.log('## use:', command.type, this.commandEmitter.listenerCount(command.type));
        this.log.debug('## has listener:', hasEventListener, hasCommandListener);
        if (!hasEventListener && !hasCommandListener) {
            this.log.warning('No listener for command', s);
        }
    }

    private handlePong(): void {
        this.missingPongs = 0;
    }

    private heartbeat(): void {
        if (this.missingPongs > 0) {
            this.log.warning('Missing pong');
        }
        if (this.missingPongs > 3) {  // TODO: make it configurable?
            this.sendAsync({ type: '_nop_' }).then(
                () => {
                    this.missingPongs = 0;
                },
                error => {
                    this.log.warning('Failed sending command; drop connection');
                    this.close(`peer lost responsive: ${util.inspect(error)}`);
                }
            );
        }
        this.missingPongs += 1;
        this.ws.ping();
    }
}
