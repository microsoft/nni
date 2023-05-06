// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  WebSocket command channel client.
 **/

import { setTimeout } from 'node:timers/promises';

import { WebSocket } from 'ws';

import { Logger, getLogger } from 'common/log';
import { WsChannel } from './channel';

// need large message because resuming experiment will re-send all final metrics in one command
const maxPayload: number = 1024 * 1024 * 1024;

export class WsChannelClient extends WsChannel {
    private logger: Logger;
    private reconnecting: boolean = false;
    private url: string;

    /**
     *  The url should start with "ws://".
     *  The name is used for better logging.
     **/
    constructor(name: string, url: string) {
        super(name);
        this.logger = getLogger(`WsChannelClient.${name}`);
        this.url = url;
        this.onLost(this.reconnect.bind(this));
    }

    public async connect(): Promise<void> {
        this.logger.debug('Connecting to', this.url);
        const ws = new WebSocket(this.url, { maxPayload });
        await this.setConnection(ws, true),
        this.logger.debug('Connected');
    }

    /**
     *  Alias of `close()`.
     **/
    public async disconnect(reason?: string): Promise<void> {
        this.close(reason ?? 'client intentionally disconnect');
    }

    private async reconnect(): Promise<void> {
        if (this.reconnecting) {
            return;
        }
        this.reconnecting = true;

        this.logger.warning('Connection lost. Try to reconnect');
        for (let i = 0; i <= 5; i++) {
            if (i > 0) {
                this.logger.warning(`Wait ${i}s before next try`);
                await setTimeout(i * 1000);
            }

            try {
                await this.connect();
                this.logger.info('Reconnect success');
                this.reconnecting = false;
                return;

            } catch (error) {
                this.logger.warning('Reconnect failed:', error);
            }
        }

        this.logger.error('Conenction lost. Cannot reconnect');
        this.emitter.emit('__error', new Error('Connection lost'));
    }
}
