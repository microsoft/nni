// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  WebSocket command channel server.
 *
 *  The server will specify a URL prefix like `ws://1.2.3.4:8080/SERVER_PREFIX`,
 *  and each client will append a channel ID, like `ws://1.2.3.4:8080/SERVER_PREFIX/CHANNEL_ID`.
 *
 *      const server = new WsChannelServer('example_server', '/SERVER_PREFIX');
 *      const url = server.getChannelUrl('CHANNEL_ID');
 *      const client = new WsChannelClient('example_client', url);
 *      await server.start();
 *      await client.connect();
 *
 *  There two styles to use the server:
 *
 *   1. Handle all clients' commands in one space:
 *
 *          server.onReceive((channelId, command) => { ... });
 *          server.send(channelId, command);
 *
 *   2. Maintain a `WsChannel` instance for each client:
 *
 *          server.onConnection((channelId, channel) => {
 *              channel.onCommand(command => { ... });
 *              channel.send(command);
 *          });
 **/

import { EventEmitter } from 'events';

import type { Request } from 'express';
import type { WebSocket } from 'ws';

import type { Command } from 'common/command_channel/interface';
import { Deferred } from 'common/deferred';
import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';
import { WsChannel } from './channel';

let heartbeatInterval: number = 5000;

type ReceiveCallback = (channelId: string, command: Command) => void;

export class WsChannelServer extends EventEmitter {
    private channels: Map<string, WsChannel> = new Map();
    private log: Logger;
    private path: string;
    private receiveCallbacks: ReceiveCallback[] = [];

    constructor(name: string, urlPath: string) {
        super();
        this.log = getLogger(`WsChannelServer.${name}`);
        this.path = urlPath;
    }

    public async start(): Promise<void> {
        const channelPath = globals.rest.urlJoin(this.path, ':channel');
        globals.rest.registerWebSocketHandler(channelPath, this.handleConnection.bind(this));
        this.log.debug('Start listening', channelPath);
    }

    public shutdown(): Promise<void> {
        const deferred = new Deferred<void>();

        this.channels.forEach((channel, channelId) => {
            channel.onClose(_reason => {
                this.channels.delete(channelId);
                if (this.channels.size === 0) {
                    deferred.resolve();
                }
            });
            channel.close('shutdown');
        });

        // wait for at most 5 seconds
        // use heartbeatInterval here for easier unit test
        setTimeout(() => {
            this.log.debug('Shutdown timeout. Stop waiting following channels:', Array.from(this.channels.keys()));
            deferred.resolve();
        }, heartbeatInterval);

        return deferred.promise;
    }

    public getChannelUrl(channelId: string, ip?: string): string {
        return globals.rest.getFullUrl('ws', ip ?? 'localhost', this.path, channelId);
    }

    public send(channelId: string, command: Command): void {
        const channel = this.channels.get(channelId);
        if (channel) {
            channel.send(command);
        } else {
            this.log.error(`Channel ${channelId} is not available`);
        }
    }

    public onReceive(callback: (channelId: string, command: Command) => void): void {
        // we configure each callback on each channel here,
        // because by this way it can detect and warning if a command is never listened
        this.receiveCallbacks.push(callback);
        for (const [channelId, channel] of this.channels) {
            channel.onReceive(command => { callback(channelId, command); });
        }
    }

    public onConnection(callback: (channelId: string, channel: WsChannel) => void): void {
        this.on('connection', callback);
    }

    private handleConnection(ws: WebSocket, req: Request): void {
        const channelId = req.params['channel'];
        this.log.debug('Incoming connection', channelId);

        if (this.channels.has(channelId)) {
            this.log.warning(`Channel ${channelId} reconnecting, drop previous connection`);
            this.channels.get(channelId)!.setConnection(ws, false);
            return;
        }

        const channel = new WsChannel(channelId);
        this.channels.set(channelId, channel);

        channel.onClose(reason => {
            this.log.debug(`Connection ${channelId} closed:`, reason);
            this.channels.delete(channelId);
        });

        channel.onError(error => {
            this.log.error(`Connection ${channelId} error:`, error);
            this.channels.delete(channelId);
        });

        for (const cb of this.receiveCallbacks) {
            channel.onReceive(command => { cb(channelId, command); });
        }

        channel.enableHeartbeat(heartbeatInterval);
        channel.setConnection(ws, false);

        this.emit('connection', channelId, channel);
    }
}

export namespace UnitTestHelpers {
    export function setHeartbeatInterval(ms: number): void {
        heartbeatInterval = ms;
    }
}
