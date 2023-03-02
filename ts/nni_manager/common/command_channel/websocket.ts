// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  WebSocket command channel server.
 *
 *  By default the server maintains a heartbeat for each client.
 *  When a client loses heartbeat for 20 seconds, the connection will be closed.
 *
 *  The constructor provides an optional parameter `robustness` to control this behavior.
 *  The parameter specifies how many consecutive missing heartbeats (at 5s interval) will kill a connection.
 *  Setting the robustness to 0 will drop dead connections in 5 seconds,
 *  and settig it to `Infinity` will effectively disable the heartbeat.
 *
 *  Note that the WebSocket connection is not reliable, even for localhost loopback connection.
 *  The client should recreate a connection when any error occurs.
 *
 *  To keep consistency with other channel types, each URL only serves one connection.
 *  When the client creates a new connection to the same URL, the previous connection will be closed by server.
 **/

import { EventEmitter } from 'events';

import type { Request } from 'express';
import type { WebSocket } from 'ws';

import { Deferred } from 'common/deferred';
import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';
import type { Command, CommandChannelServer } from './interface';

let heartbeatInterval: number = 5000;

type ReceiveCallback = (channelId: string, command: Command) => void;

export class WebSocketChannelServer implements CommandChannelServer {
    private connectionCount: number = 0;
    private connections: Map<string, Connection> = new Map();
    private log: Logger;
    private path: string;
    private receiveCallbacks: ReceiveCallback[] = [];
    private robustness: number = 3;

    constructor(name: string, urlPath: string, robustness?: number) {
        this.log = getLogger(`WebSocketChannelServer.${name}`);
        this.path = urlPath;
        if (robustness !== undefined) {
            this.robustness = robustness;
        }
    }

    public async start(): Promise<void> {
        const channelPath = globals.rest.urlJoin(this.path, ':channel');
        globals.rest.registerWebSocketHandler(channelPath, this.handleConnection.bind(this));
    }

    public shutdown(): Promise<void> {
        const deferred = new Deferred<void>();

        this.connections.forEach((conn, channelId) => {
            conn.removeAllListeners();
            conn.on('close', (_code, _reason) => {
                this.connections.delete(channelId);
                if (this.connections.size === 0) {
                    deferred.resolve();
                }
            });
            conn.close('shutdown');
        });

        setTimeout(() => {
            deferred.resolve();
        }, heartbeatInterval);

        return deferred.promise;
    }

    public getChannelUrl(channelId: string): string {
        return globals.rest.getFullUrl('ws', this.path, channelId);
    }

    public send(channelId: string, command: Command): void {
        const conn = this.connections.get(channelId);
        if (conn === undefined) {
            this.log.error(`Channel ${channelId} is not available`);
        } else {
            conn.send(command);
        }
    }

    public onReceive(callback: (channelId: string, command: Command) => void): void {
        this.receiveCallbacks.push(callback);
    }

    private handleConnection(ws: WebSocket, req: Request): void {
        const channelId = req.params['channel'];
        this.connectionCount += 1;
        const connName = `${channelId}#${this.connectionCount}`;
        this.log.debug('Incoming connection', connName);

        if (this.connections.has(channelId)) {
            this.log.warning(`Channel ${channelId} reconnecting, drop previous connection`);
            this.connections.get(channelId)!.close('reconnect');
        }

        const conn = new Connection(connName, ws, this.robustness);
        this.connections.set(channelId, conn);

        conn.on('close', (code, reason) => {
            this.log.debug(`Connection ${connName} closed:`, code, reason);
            this.connections.delete(channelId);
        });

        conn.on('error', error => {
            this.log.error(`Connection ${connName} error:`, error);
            this.connections.delete(channelId);
        });

        conn.on('receive', command => {
            for (const cb of this.receiveCallbacks) {
                cb(channelId, command);
            }
        });
    }
}

class Connection extends EventEmitter {
    private closing: boolean = false;
    private heartbeatTimer: NodeJS.Timer;
    private log: Logger;
    private missingPongs: number = 0;
    private robustness: number;
    private ws: WebSocket;

    constructor(name: string, ws: WebSocket, robustness: number) {
        super();
        this.log = getLogger(`WebSocketChannelManager.Connection.${name}`);
        this.ws = ws;
        this.robustness = robustness;
        ws.on('close', this.handleClose.bind(this));
        ws.on('error', this.handleError.bind(this));
        ws.on('message', this.handleMessage.bind(this));
        ws.on('pong', this.handlePong.bind(this));
        this.heartbeatTimer = setInterval(this.heartbeat.bind(this), heartbeatInterval);
    }

    public send(command: Command): void {
        this.log.trace('Sending command', command);
        this.ws.send(JSON.stringify(command));
    }

    public close(reason: string): void {
        this.log.debug('Closing');
        this.closing = true;
        clearInterval(this.heartbeatTimer);
        this.ws.close(1001, reason);
    }

    private handleClose(code: number, reason: Buffer): void {
        if (this.closing) {
            this.log.debug('Connection closed');
        } else {
            this.log.debug('Client closed connection:', code, String(reason));
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
        if (this.closing) {
            this.log.warning('Received message after closing:', String(data));
        } else {
            const s = String(data);
            this.log.trace('Received command', s);
            this.emit('receive', JSON.parse(s));
        }
    }

    private handlePong(): void {
        this.missingPongs = 0;
    }

    private heartbeat(): void {
        if (this.missingPongs > 0) {
            this.log.warning('Client loses responsive');
        }
        if (this.missingPongs > this.robustness) {
            this.log.warning('Drop connection');
            this.close('no pong');
        }
        this.missingPongs += 1;
        this.ws.ping();
    }
}

export namespace UnitTestHelpers {
    export function setHeartbeatInterval(ms: number): void {
        heartbeatInterval = ms;
    }
}
