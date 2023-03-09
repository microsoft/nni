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

import type { Command, CommandChannelServer } from 'common/command_channel/interface';
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
            channel.on('close', (_reason) => {
                this.channels.delete(channelId);
                if (this.channels.size === 0) {
                    deferred.resolve();
                }
            });
            channel.close('shutdown');
        });

        // wait for at most 5 seconds
        // use heartbeatInterval here for easier unit test
        setTimeout(() => { deferred.resolve(); }, heartbeatInterval);

        return deferred.promise;
    }

    public getChannelUrl(channelId: string): string {
        return globals.rest.getFullUrl('ws', this.path, channelId);
    }

    public send(channelId: string, command: Command): void {
        const channel = this.channels.get(channelId);
        if (channel === undefined) {
            this.log.error(`Channel ${channelId} is not available`);
        } else {
            channel.send(command);
        }
    }

    public onReceive(callback: (channelId: string, command: Command) => void): void {
        // we configure each callback on each channel here,
        // because by this way it can detect and warning if a command is never listened
        this.receiveCallbacks.push(callback);
        for (const [channelId, channel] of this.channels) {
            channel.onCommand(command => { callback(channelId, command); });
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
            this.channels.get(channelId)!.setConnection(ws);
            return;
        }

        const channel = new WsChannel(channelId, ws, heartbeatInterval);
        this.channels.set(channelId, channel);

        channel.on('close', reason => {
            this.log.debug(`Connection ${channelId} closed:`, reason);
            this.channels.delete(channelId);
        });

        channel.on('error', error => {
            this.log.error(`Connection ${channelId} error:`, error);
            this.channels.delete(channelId);
        });

        for (const cb of this.receiveCallbacks) {
            channel.on('command', command => { cb(channelId, command); });
        }

        this.emit('connection', channelId, channel);
    }
}

export namespace UnitTestHelpers {
    export function setHeartbeatInterval(ms: number): void {
        heartbeatInterval = ms;
    }
}
