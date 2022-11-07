// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { Response, Router } from 'express';

import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';

const logger = getLogger('staging');

const sharedEnvRouter: Router = Router();

export type Command = any;

export function getEnvironmentRouter(): Router {
    return sharedEnvRouter;
}

export class HttpChannelManager {
    private consumerQueues: Map<string, Deferred<Command>[]> = new Map();
    private producerQueues: Map<string, Command[]> = new Map();
    private receiveCallback: ((channelId: string, command: Command) => void) | null = null;

    constructor(environmentId: string) {
        sharedEnvRouter.put(`/${environmentId}/:channel`, (req, res) => {
            const channelId = req.params['channel']
            logger.debug('PUT', channelId, req.body);
            if (this.receiveCallback === null) {
                logger.error('No receive callback:', channelId, req.body);
                return;
            }
            this.receiveCallback(channelId, req.body);
            res.send();
        });

        sharedEnvRouter.get(`/${environmentId}/:channel`, (req, res) => {
            const channelId = req.params['channel'];
            logger.debug('GET', channelId);
            const command = this.popProducer(channelId);
            if (command === null) {
                this.pushConsumer(channelId, this.newConsumer(res));
            } else {
                logger.debug('send', command);
                res.send(command);
            }
        });
    }

    public onReceive(callback: (channelId: string, command: Command) => void): void {
        if (this.receiveCallback !== null) {
            logger.error('Multiple receive callbacks');
            return;
        }
        this.receiveCallback = callback;
    }

    public send(channelId: string, command: Command): void {
        const consumer = this.popConsumer(channelId);
        if (consumer === null) {
            this.pushProducer(channelId, command);
        } else {
            consumer.resolve(command);
        }
    }

    private newConsumer(res: Response): Deferred<Command> {
        const consumer = new Deferred<Command>();
        consumer.promise.then(
            command => { logger.debug('send', command); res.send(command); },
            _reason => { logger.debug('send status 408'); res.sendStatus(408); }
        );
        setTimeout(() => { consumer.reject(new Error('timeout')); }, 1000);
        return consumer;
    }

    private pushConsumer(channelId: string, consumer: Deferred<Command>) {
        const queue = this.consumerQueues.get(channelId);
        if (queue === undefined) {
            this.consumerQueues.set(channelId, [ consumer ]);
        } else {
            queue.push(consumer);
        }
    }

    private popConsumer(channelId: string): Deferred<Command> | null {
        const queue = this.consumerQueues.get(channelId);
        if (queue && queue.length > 0) {
            return queue.shift()!;
        } else {
            return null;
        }
    }

    private pushProducer(channelId: string, producer: Command): void {
        const queue = this.producerQueues.get(channelId);
        if (queue === undefined) {
            this.producerQueues.set(channelId, [ producer ]);
        } else {
            queue.push(producer);
        }
    }

    private popProducer(channelId: string): Command | null {
        const queue = this.producerQueues.get(channelId);
        if (queue && queue.length > 0) {
            return queue.shift();
        } else {
            return null;
        }
    }
}
