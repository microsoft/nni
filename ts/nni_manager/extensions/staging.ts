// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Staging APIs for web portal development.
 *
 *  Usage:
 *
 *      Create connections to "ws://localhost:8080/staging" or "ws://localhost:8080/staging/CHANNEL".
 *
 *      The connections will send a message per second to the each clients.
 *      If CHANNEL is a number, the interval will be changed from 1s to CHANNEL seconds.
 *
 *      Each message contains an incremental counter.
 *      The counter should be independent for each channel,
 *      and should be synchronized for all connections on a same channel.
 *
 *      The clients can send arbitrary messages to the server.
 *      If the message is a non-negative number, the counter (of corresponding channel) will be reset to the number.
 *      If the message is a negative number, all connections to the channel will be closed.
 **/

import type { WebSocket } from 'ws';

import { Deferred } from 'common/deferred';
import { globals } from 'common/globals';
import { getLogger } from 'common/log';

const logger = getLogger('Staging');

export function enableWebuiStaging(): void {
    logger.debug('enabled');
    new WsStaging();
}

class WsStaging {
    private conns: Map<string, Conn> = new Map();

    constructor() {
        globals.rest.registerWebSocketHandler('/staging', (ws, _req) => {
            this.acceptConnection(ws);
        });
        globals.rest.registerWebSocketHandler('/staging/:channel', (ws, req) => {
            this.acceptConnection(ws, req.params['channel']);
        });

        globals.shutdown.register('Staging', this.shutdown.bind(this));
    }

    public shutdown(): Promise<void> {
        const deferred = new Deferred<void>();

        this.conns.forEach(conn => {
            conn.sockets.forEach(ws => {
                ws?.on('close', () => {
                    this.conns.delete(conn.channel);
                    if (this.conns.size === 0) {
                        deferred.resolve();
                    }
                });
                ws?.close(4001, 'shutdown');
            });
        });

        setTimeout(() => { deferred.resolve(); }, 5000);
        return deferred.promise;
    }

    private acceptConnection(ws: WebSocket, channel: string = '_main_'): void {
        logger.debug('connect:', channel);

        const conn = this.conns.get(channel) ?? new Conn(ws, channel);
        this.conns.set(channel, conn);
        const wsIdx = conn.sockets.length - 1;

        const interval = (Number(channel) || 1) * 1000;
        conn.timer = setInterval(() => {
            if (conn.count < 0) {  // close
                clearInterval(conn.timer);
                conn.sockets.forEach(ws => { ws?.close(4000, 'negative counter'); });
                this.conns.delete(channel);
            } else {
                const msg = JSON.stringify({ channel: conn.channel, count: conn.count });
                conn.count += 1;
                conn.sockets.forEach(ws => { ws?.send(msg); });
            }
        }, interval);

        ws.on('close', (code, reason) => {
            logger.debug('close by client:', channel, code, reason);
            conn.sockets[wsIdx] = null;
        });

        ws.on('error', (error) => {
            logger.debug('error:', channel, error);
            conn.sockets[wsIdx] = null;
        });

        ws.on('message', (data, _isBinary) => {
            const s = String(data);
            logger.debug('message:', channel, s);
            const n = Number(s);
            if (!isNaN(n)) {
                conn.count = n;
            }
        });
    }
}

class Conn {
    public channel: string;
    public sockets: (WebSocket | null)[];
    public count: number = 0;
    public timer!: NodeJS.Timer;

    constructor(ws: WebSocket, channel: string) {
        this.channel = channel;
        this.sockets = [ ws ];
    }
}
