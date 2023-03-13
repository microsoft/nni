// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  WebSocket command channel.
 *
 *  This is the base class that used by both server and client.
 *
 *  For the server, channels can be got with `onConnection()` event listener.
 *  For the client, a channel can be created with `new WsChannelClient()` subclass.
 *  Do not use the constructor directly.
 *
 *  The channel is fault tolerant to some extend. It has three different types of closing related events:
 *
 *   1. "close": The channel is intentionally closed.
 *
 *      This is caused either by "close()" or "disconnect()" call, or by receiving a "_bye_" command from the peer.
 *
 *   2. "lost": The channel is temporarily unavailable and is trying to recover.
 *      (The high level class should examine the peer's status out-of-band when receiving this event.)
 *
 *      When the underlying socket is dead, this event is emitted.
 *      The client will try to reconnect in around 15s. If all attempts fail, an "error" event will be emitted.
 *      The server will wait the client for 30s. If it does not reconnect, an "error" event will be emitted.
 *      Successful recover will not emit command.
 *
 *   3. "error": The channel is dead and cannot recover.
 *
 *      A "close" event may or may not follow this event. Do not rely on that.
 **/

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
    private closing: boolean = false;
    private commandEmitter: EventEmitter = new EventEmitter();
    private connection: WsConnection | null = null;
    private epoch: number = 0;
    private heartbeatInterval: number | null;
    private log: Logger;

    public readonly name: string;

    // internal, don't use
    constructor(name: string, ws?: WebSocket, heartbeatInterval?: number) {
        super()
        this.log = getLogger(`WsChannel.${name}`);
        this.name = name;
        this.heartbeatInterval = heartbeatInterval || null;
        if (ws) {
            this.setConnection(ws);
        }
    }

    public enableHeartbeat(interval: number): void {
        this.heartbeatInterval = interval;
    }

    // internal, don't use
    public setConnection(ws: WebSocket): void {
        if (this.connection) {
            this.log.debug('Abandon previous connection');
            this.epoch += 1;
        }
        this.log.debug(`Epoch ${this.epoch} start`);
        this.connection = this.configConnection(ws);
    }

    public close(reason: string): void {
        this.log.debug('Close channel:', reason);
        if (this.connection) {
            this.connection.close(reason);
            this.endEpoch();
        }
        if (!this.closing) {
            this.closing = true;
            this.emit('close', reason);
        }
    }

    public send(command: Command): void {
        if (this.connection) {
            this.connection.send(command);
        } else {
            // TODO: add a queue?
            this.log.error('Connection lost. Dropped command', command);
        }
    }

    /**
     *  Async version of `send()` that (partially) ensures the command is successfully sent to peer.
     **/
    public sendAsync(command: Command): Promise<void> {
        if (this.connection) {
            return this.connection.sendAsync(command);
        } else {
            this.log.error('Connection lost. Dropped command async', command);
            return Promise.reject(new Error('Connection is lost and trying to recover, cannot send command now'));
        }
    }

    // the first overload listens to all commands, while the second listens to one command type
    public onCommand(callback: (command: Command) => void): void;
    public onCommand(commandType: string, callback: (command: Command) => void): void;

    public onCommand(commandTypeOrCallback: any, callbackOrNone?: any): void {
        if (callbackOrNone) {
            this.commandEmitter.on(commandTypeOrCallback, callbackOrNone);
        } else {
            this.commandEmitter.on('__any', commandTypeOrCallback);
        }
    }

    private configConnection(ws: WebSocket): WsConnection {
        const epoch = this.epoch;  // copy it to use in closure
        const conn = new WsConnection(
            this.epoch ? `${this.name}.${epoch}` : this.name,
            ws,
            this.commandEmitter,
            this.heartbeatInterval
        );

        conn.on('bye', reason => {
            this.log.debug('Peer intentionally close:', reason);
            this.endEpoch();
            if (!this.closing) {
                this.closing = true;
                this.emit('close', reason);
            }
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
        if (this.closing) {
            this.log.debug('Connection cleaned up:', reason);
            return;
        }
        if (this.epoch !== epoch) {  // the connection is already abandoned
            this.log.debug(`Previous connection closed ${epoch}: ${reason}`);
            return;
        }

        this.log.warning('Connection closed unexpectedly:', reason);
        this.emit('lost');
        this.endEpoch();
        // the reconnect logic is in client subclass
    }

    private endEpoch(): void {
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

    public async close(reason: string): Promise<void> {
        if (this.closing) {
            this.log.debug('Close again:', reason);
            return;
        }

        this.log.debug('Close:', reason);
        this.closing = true;
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }

        try {
            await this.sendAsync({ type: '_bye_', reason });
        } catch (error) {
            this.log.error('Failed to send bye:', error);
        }

        try {
            this.ws.close(1001, reason);
        } catch (error) {
            this.log.error('Failed to close:', error);
            this.ws.terminate();
        }
    }

    private terminate(reason: string): void {
        this.log.debug('Terminate:', reason);
        this.closing = true;
        try {
            this.ws.close(1001, reason);
        } catch (error) {
            this.log.debug('Failed to close:', error);
            this.ws.terminate();
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

        const hasAnyListener = this.commandEmitter.emit('__any', command);
        const hasTypeListener = this.commandEmitter.emit(command.type, command);
        if (!hasAnyListener && !hasTypeListener) {
            this.log.warning('No listener for command', s);
        }
    }

    private handlePong(): void {
        this.log.debug('receive pong'); // todo
        this.missingPongs = 0;
    }

    private heartbeat(): void {
        if (this.missingPongs > 0) {
            this.log.warning('Missing pong');
        }
        if (this.missingPongs > 3) {  // TODO: make it configurable?
            // no response for ping, try real command
            this.sendAsync({ type: '_nop_' }).then(() => {
                this.missingPongs = 0;
            }).catch(error => {
                this.log.error('Failed sending command. Drop connection:', error);
                this.terminate(`peer lost responsive: ${util.inspect(error)}`);
            });
        }
        this.missingPongs += 1;
        this.log.debug('send ping');
        this.ws.ping();
    }
}
