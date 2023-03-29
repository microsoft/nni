// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Internal helper class which handles one WebSocket connection.
 **/

import { EventEmitter } from 'node:events';
import util from 'node:util';

import type { WebSocket } from 'ws';

import type { Command } from 'common/command_channel/interface';
import { Logger, getLogger } from 'common/log';

interface ConnectionEvents {
    'bye': (reason: string) => void;
    'close': (code: number, reason: string) => void;
    'error': (error: Error) => void;
}

export declare interface WsConnection {
    on<E extends keyof ConnectionEvents>(event: E, listener: ConnectionEvents[E]): this;
}

export class WsConnection extends EventEmitter {
    private closing: boolean = false;
    private commandEmitter: EventEmitter;
    private heartbeatTimer: NodeJS.Timer | null = null;
    private log: Logger;
    private missingPongs: number = 0;

    public readonly ws: WebSocket;

    constructor(name: string, ws: WebSocket, commandEmitter: EventEmitter) {
        super();
        this.log = getLogger(`WsConnection.${name}`);
        this.ws = ws;
        this.commandEmitter = commandEmitter;

        ws.on('close', this.handleClose.bind(this));
        ws.on('error', this.handleError.bind(this));
        ws.on('message', this.handleMessage.bind(this));
        ws.on('pong', this.handlePong.bind(this));
    }

    public setHeartbeatInterval(interval: number): void {
        if (this.heartbeatTimer) {
            clearTimeout(this.heartbeatTimer);
        }
        this.heartbeatTimer = setInterval(this.heartbeat.bind(this), interval);
    }

    public async close(reason: string): Promise<void> {
        if (this.closing) {
            this.log.debug('Close again:', reason);
            return;
        }

        this.log.debug('Close connection:', reason);
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
            this.ws.close(4000, reason);
            return;
        } catch (error) {
            this.log.error('Failed to close socket:', error);
        }

        try {
            this.ws.terminate();
        } catch (error) {
            this.log.debug('Failed to terminate socket:', error);
        }
    }

    public terminate(reason: string): void {
        this.log.debug('Terminate connection:', reason);
        this.closing = true;
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }

        try {
            this.ws.close(4001, reason);
            return;
        } catch (error) {
            this.log.debug('Failed to close socket:', error);
        }

        try {
            this.ws.terminate();
        } catch (error) {
            this.log.debug('Failed to terminate socket:', error);
        }
    }

    public send(command: Command): void {
        this.log.trace('Send command', command);
        this.ws.send(JSON.stringify(command));
    }

    public sendAsync(command: Command): Promise<void> {
        this.log.trace('(async) Send command', command);
        const send: any = util.promisify(this.ws.send.bind(this.ws));
        return send(JSON.stringify(command));
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
            this.log.error('Connection error:', error);
            this.emit('error', error);
        }
    }

    private handleMessage(data: Buffer, _isBinary: boolean): void {
        const s = String(data);
        if (this.closing) {
            this.log.warning('Received message after closing:', s);
            return;
        }

        this.log.trace('Receive command', s);
        const command = JSON.parse(s);

        if (command.type === '_nop_') {
            return;
        }

        if (command.type === '_bye_') {
            this.log.debug('Intentionally close connection:', s);
            this.closing = true;
            this.emit('bye', command.reason);
            return;
        }

        const hasReceiveListener = this.commandEmitter.emit('__receive', command);
        const hasCommandListener = this.commandEmitter.emit(command.type, command);
        if (!hasReceiveListener && !hasCommandListener) {
            this.log.warning('No listener for command', s);
        }
    }

    private handlePong(): void {
        this.log.trace('Receive pong');
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
        this.log.trace('Send ping');
        this.ws.ping();
    }
}
