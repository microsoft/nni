// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  HTTP based command channel. Inefficient but simple.
 *
 *  Client to server command is implemented by naive PUT request.
 *
 *  Server to client command is implemented by polling with GET requests.
 *  The client should repeat GET requests without intervals when it is receiving commands.
 *  If the server has an outgoing command, it will be sent as response.
 *  If there is no command, the server will wait for 1 second and response 408 "Request Timeout" status code.
 **/

import { EventEmitter } from 'events';

import { Request, Response } from 'express';

import { Deferred } from 'common/deferred';
import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';
import type { Command, CommandChannelServer } from './interface';

let timeoutMilliseconds = 1000;

const HttpRequestTimeout = 408;
const HttpGone = 410;

export class HttpChannelServer implements CommandChannelServer {
    private emitter: EventEmitter = new EventEmitter();
    private log: Logger;
    // the server can only send commands when the client requests, so it needs a queue
    private outgoingQueues: Map<string, CommandQueue> = new Map();
    private path: string;
    private serving: boolean = false;

    constructor(name: string, urlPath: string) {
        this.log = getLogger(`HttpChannelManager.${name}`);
        this.path = urlPath;
    }

    public async start(): Promise<void> {
        this.serving = true;
        const channelPath = globals.rest.urlJoin(this.path, ':channel');
        globals.rest.registerSyncHandler('GET', channelPath, this.handleGet.bind(this));
        globals.rest.registerSyncHandler('PUT', channelPath, this.handlePut.bind(this));
    }

    public async shutdown(): Promise<void> {
        this.serving = false;
        this.outgoingQueues.forEach(queue => { queue.clear(); });
    }

    public getChannelUrl(channelId: string, ip?: string): string {
        return globals.rest.getFullUrl('http', ip ?? 'localhost', this.path, channelId);
    }

    public send(channelId: string, command: Command): void {
        this.getOutgoingQueue(channelId).push(command);
    }

    public onReceive(callback: (channelId: string, command: Command) => void): void {
        this.emitter.on('receive', callback);
    }

    public onConnection(_callback: (channelId: string, channel: any) => void): void {
        throw new Error('Not implemented');
    }

    private handleGet(request: Request, response: Response): void {
        const channelId = request.params['channel'];
        const promise = this.getOutgoingQueue(channelId).asyncPop(timeoutMilliseconds);
        promise.then(command => {
            if (command === null) {
                response.sendStatus(this.serving ? HttpRequestTimeout : HttpGone);
            } else {
                response.send(command);
            }
        });
    }

    private handlePut(request: Request, response: Response): void {
        if (!this.serving) {
            response.sendStatus(HttpGone);
            return;
        }

        const channelId = request.params['channel'];
        const command = request.body;
        this.emitter.emit('receive', channelId, command);
        response.send();
    }

    private getOutgoingQueue(channelId: string): CommandQueue {
        if (!this.outgoingQueues.has(channelId)) {
            this.outgoingQueues.set(channelId, new CommandQueue());
        }
        return this.outgoingQueues.get(channelId)!;
    }
}

/**
 *  A FIFO queue with asynchronous pop.
 **/
class CommandQueue {
    private commands: Command[] = [];
    private consumers: Deferred<Command | null>[] = [];

    public push(command: Command): void {
        const consumer = this.consumers.shift();
        if (consumer !== undefined) {
            consumer.resolve(command);
        } else {
            this.commands.push(command);
        }
    }

    /**
     *  The returned promise will be resolved when there is a command available,
     *  or it will be resolved with null if `timeout` milliseconds passed before getting a command.
     *
     *  That means, if there is already a command in the queue, the promise is intermediately resolved;
     *  if the queue is empty, it will be resolved either at a corresponding `push()` call,
     *  or at `timeout` ms after this call.
     **/
    public asyncPop(timeout: number): Promise<Command | null> {
        const command = this.commands.shift();
        if (command !== undefined) {
            return Promise.resolve(command);
        } else {
            const consumer = new Deferred<Command | null>();
            this.consumers.push(consumer);
            setTimeout(() => {
                if (!consumer.settled) {
                    this.consumers = this.consumers.filter(item => (item !== consumer));
                    consumer.resolve(null);
                }
            }, timeout);
            return consumer.promise;
        }
    }

    /**
     *  Make all pending promises timeout (resolve with null).
     **/
    public clear(): void {
        for (const consumer of this.consumers) {
            consumer.resolve(null);
        }
    }
}

export namespace UnitTestHelpers {
    export function setTimeoutMilliseconds(ms: number): void {
        timeoutMilliseconds = ms;
    }
}
