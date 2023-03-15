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

import { EventEmitter, once } from 'node:events';
import util from 'node:util';

import type { WebSocket } from 'ws';

import type { Command, CommandChannel } from 'common/command_channel/interface';
import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';
import { Connection } from './connection';

interface QueuedCommand {
    command: Command;
    deferred?: Deferred<void>;
}

export class WsChannel implements CommandChannel {
    private closing: boolean = false;
    private connection: Connection | null = null;
    private epoch: number = -1;
    private heartbeatInterval: number | null = null;
    private log: Logger;
    private queue: QueuedCommand[] = [];

    protected emitter: EventEmitter = new EventEmitter();

    public readonly name: string;

    // internal, don't use
    constructor(name: string) {
        this.log = getLogger(`WsChannel.${name}`);
        this.name = name;
    }

    // internal, don't use
    // must be called after enableHeartbeat()
    public async setConnection(ws: WebSocket, waitOpen: boolean): Promise<void> {
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
        this.heartbeatInterval = interval ?? 5000;
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

    private newEpoch(): void {
        this.connection = null;
        this.epoch += 1;
    }

    private configConnection(ws: WebSocket): Connection {
        const connName = this.epoch ? `${this.name}.${this.epoch}` : this.name;
        const conn = new Connection(connName, ws, this.emitter, this.heartbeatInterval);

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

    private dropConnection(conn: Connection, reason: string): void {
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
        // the reconnect logic is in client subclass
    }
}
