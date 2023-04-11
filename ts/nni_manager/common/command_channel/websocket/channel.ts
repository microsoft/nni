// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  WebSocket command channel.
 *
 *  A WsChannel operates on one WebSocket connection at a time.
 *  But when the network is unstable, it may close the underlying connection and create a new one.
 *  This is generally transparent to the user of this class, except that a "lost" event will be emitted.
 *
 *  To distinguish intentional close from connection lost,
 *  a "_bye_" command will be sent when `close()` or `disconnect()` is invoked.
 *
 *  If the connection is closed before receiving "_bye_" command, a "lost" event will be emitted and:
 *
 *    * The client will try to reconnect for severaly times in around 15s.
 *    * The server will wait the client to reconnect for around 30s.
 *
 *  If the reconnecting attempt failed, both side will emit an "error" event.
 **/

import { EventEmitter, once } from 'node:events';
import util from 'node:util';

import type { WebSocket } from 'ws';

import type { Command, CommandChannel } from 'common/command_channel/interface';
import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';
import { WsConnection } from './connection';

interface QueuedCommand {
    command: Command;
    deferred?: Deferred<void>;
}

export class WsChannel implements CommandChannel {
    private closing: boolean = false;
    private connection: WsConnection | null = null;  // NOTE: used in unit test
    private epoch: number = -1;
    private heartbeatInterval: number | null = null;
    private log: Logger;
    private queue: QueuedCommand[] = [];
    private terminateTimer: NodeJS.Timer | null = null;

    protected emitter: EventEmitter = new EventEmitter();

    public readonly name: string;

    // internal, don't use
    constructor(name: string) {
        this.log = getLogger(`WsChannel.${name}`);
        this.name = name;
    }

    // internal, don't use
    public async setConnection(ws: WebSocket, waitOpen: boolean): Promise<void> {
        if (this.terminateTimer) {
            clearTimeout(this.terminateTimer);
            this.terminateTimer = null;
        }

        this.connection?.terminate('new epoch start');
        this.newEpoch();
        this.log.debug(`Epoch ${this.epoch} start`);

        this.connection = this.configConnection(ws);
        if (waitOpen) {
            await once(ws, 'open');
        }

        while (this.connection && this.queue.length > 0) {
            const item = this.queue.shift()!;
            try {
                await this.connection.sendAsync(item.command);
                item.deferred?.resolve();
            } catch (error) {
                this.log.error('Failed to send command on recovered channel:', error);
                this.log.error('Dropped command:', item.command);
                item.deferred?.reject(error as any);
                // it should trigger connection's error event and this.connection will be set to null
            }
        }
    }

    public enableHeartbeat(interval?: number): void {
        this.heartbeatInterval = interval ?? defaultHeartbeatInterval;
        this.connection?.setHeartbeatInterval(this.heartbeatInterval);
    }

    public close(reason: string): void {
        this.log.debug('Close channel:', reason);
        this.connection?.close(reason);
        if (this.setClosing()) {
            this.emitter.emit('__close', reason);
        }
    }

    public terminate(reason: string): void {
        this.log.info('Terminate channel:', reason);
        this.connection?.terminate(reason);
        if (this.setClosing()) {
            this.emitter.emit('__error', new Error(`WsChannel terminated: ${reason}`));
        }
    }

    public send(command: Command): void {
        if (this.closing) {
            this.log.error('Channel closed. Ignored command', command);
            return;
        }

        if (!this.connection) {
            this.log.warning('Connection lost. Enqueue command', command);
            this.queue.push({ command });
            return;
        }

        this.connection.send(command);
    }

    public sendAsync(command: Command): Promise<void> {
        if (this.closing) {
            this.log.error('(async) Channel closed. Refused command', command);
            return Promise.reject(new Error('WsChannel has been closed'));
        }

        if (!this.connection) {
            this.log.warning('(async) Connection lost. Enqueue command', command);
            const deferred = new Deferred<void>();
            this.queue.push({ command, deferred });
            return deferred.promise;
        }

        return this.connection.sendAsync(command);
    }

    public onReceive(callback: (command: Command) => void): void {
        this.emitter.on('__receive', callback);
    }

    public onCommand(commandType: string, callback: (command: Command) => void): void {
        this.emitter.on(commandType, callback);
    }

    public onClose(callback: (reason?: string) => void): void {
        this.emitter.on('__close', callback);
    }

    public onError(callback: (error: Error) => void): void {
        this.emitter.on('__error', callback);
    }

    public onLost(callback: () => void): void {
        this.emitter.on('__lost', callback);
    }

    // TODO: temporary api for tuner command channel
    public getBufferedAmount(): number {
        return this.connection?.ws.bufferedAmount ?? 0;
    }

    private newEpoch(): void {
        this.connection = null;
        this.epoch += 1;
    }

    private configConnection(ws: WebSocket): WsConnection {
        const connName = this.epoch ? `${this.name}.${this.epoch}` : this.name;
        const conn = new WsConnection(connName, ws, this.emitter);
        if (this.heartbeatInterval) {
            conn.setHeartbeatInterval(this.heartbeatInterval);
        }

        conn.on('bye', reason => {
            this.log.debug('Peer intentionally closing:', reason);
            if (this.setClosing()) {
                this.emitter.emit('__close', reason);
            }
        });

        conn.on('close', (code, reason) => {
            this.log.debug('Peer closed:', reason);
            this.dropConnection(conn, `Peer closed: ${code} ${reason}`);
        });

        conn.on('error', error => {
            this.dropConnection(conn, `Connection error: ${util.inspect(error)}`);
        });

        return conn;
    }

    private setClosing(): boolean {
        if (this.closing) {
            return false;
        }
        this.closing = true;
        this.newEpoch();
        this.queue.forEach(item => {
            item.deferred?.reject(new Error('WsChannel has been closed.'));
        });
        return true;
    }

    private dropConnection(conn: WsConnection, reason: string): void {
        if (this.closing) {
            this.log.debug('Clean up:', reason);
            return;
        }
        if (this.connection !== conn) {  // the connection is already abandoned
            this.log.debug(`Previous connection closed: ${reason}`);
            return;
        }

        this.log.warning('Connection closed unexpectedly:', reason);
        this.newEpoch();
        this.emitter.emit('__lost');

        if (!this.terminateTimer) {
            this.terminateTimer = setTimeout(() => {
                if (!this.closing) {
                    this.terminate('have not reconnected in 30s');
                }
            }, terminateTimeout);
        }

        // the reconnect logic is in client subclass
    }
}

let defaultHeartbeatInterval: number = 5000;
let terminateTimeout: number = 30000;

export namespace UnitTestHelper {
    export function setHeartbeatInterval(ms: number): void {
        defaultHeartbeatInterval = ms;
    }

    export function setTerminateTimeout(ms: number): void {
        terminateTimeout = ms;
    }

    export function reset(): void {
        defaultHeartbeatInterval = 5000;
        terminateTimeout = 30000;
    }
}
