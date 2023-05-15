// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Staging APIs for web portal development.
 **/

import type { WebSocket } from 'ws';

import { Deferred } from 'common/deferred';
import { globals } from 'common/globals';
import { getLogger } from 'common/log';

const logger = getLogger('Staging');

export function enableWebuiStaging(): void {
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
            conn.ws.on('close', () => {
                this.conns.delete(conn.channel);
                if (this.conns.size === 0) {
                    deferred.resolve();
                }
            });
            conn.ws.close(4000, 'shutdown');
        });

        setTimeout(() => { deferred.resolve(); }, 5000);
        return deferred.promise;
    }

    private acceptConnection(ws: WebSocket, channel: string = '_main_'): void {
        logger.debug('connect:', channel);

        const conn = new Conn(ws, channel);
        this.conns.set(channel, conn);

        const interval = (Number(channel) || 1) * 1000;
        conn.timer = setInterval(() => {
            if (conn.count < 0) {  // close
                clearInterval(conn.timer);
                this.conns.delete(channel);
            } else {
                const msg = { channel: conn.channel, count: conn.count };
                conn.ws.send(JSON.stringify(msg));
            }
        }, interval);

        ws.on('close', (code, reason) => {
            logger.debug('close by client:', channel, code, reason);
            conn.count = -1;
        });

        ws.on('error', (error) => {
            logger.debug('error:', channel, error);
            conn.count = -1;
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
    channel: string;
    ws: WebSocket;
    count: number = 0;
    timer!: NodeJS.Timer;

    constructor(ws: WebSocket, channel: string) {
        this.ws = ws;
        this.channel = channel;
    }
}
