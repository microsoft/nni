// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { setTimeout } from 'node:timers/promises';

import { WebSocket } from 'ws';

import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';
import { WsChannel } from './channel';

const maxPayload: number = 4 * 1024 * 1024 * 1024;

export class WsChannelClient extends WsChannel {
    private logger: Logger;  // avoid name conflict with base class
    private url: string;

    constructor(url: string, name?: string) {
        const name_ = name ?? generateName(url);
        super(name_);
        this.logger = getLogger(`WsClient.${name_}`);
        this.logger.trace('Created with URL:', url);
        this.url = url;
        this.on('lost', this.reconnect.bind(this));
    }

    public async connect(): Promise<void> {
        this.logger.debug('Connecting to', this.url);
        const ws = new WebSocket(this.url, { maxPayload });

        // fixme
        this.setConnection(ws);

        const deferred = new Deferred<void>();
        ws.once('open', () => {
            deferred.resolve();
        });
        ws.once('error', error => {
            deferred.reject(error);
        });
        await deferred.promise;

        this.logger.debug('Connected');
    }

    public disconnect(reason?: string): Promise<void> {
        return this.close(reason ?? 'client disconnecting');
    }

    private async reconnect(): Promise<void> {
        this.logger.warning('Connection lost; try to reconnect');
        for (let i = 0; i <= 5; i++) {
            if (i > 0) {
                this.logger.warning(`Wait ${i}s before next try`);
                await setTimeout(i * 1000);
            }

            try {
                await this.connect();
                this.logger.info('Reconnect success');
                return;
            } catch (error) {
                this.logger.warning('Reconnect failed:', error);
            }
        }

        this.logger.error('Conenction lost; cannot reconnect');
        this.emit('error', new Error('Connection lost'));
    }
}

function generateName(url: string): string {
    const parts = url.split('/');
    for (let i = parts.length - 1; i > 1; i--) {
        if (parts[i]) {
            return parts[i];
        }
    }
    return 'anonymous';
}
