// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  WebSocket command channel client.
 *
 *  Usage:
 *
 *      const client = new WsChannelClient('ws://1.2.3.4:8080/server/channel_id');
 *      await client.connect();
 *      client.send(command);
 *
 *  Most APIs are derived the base class `WsChannel`.
 *  See its doc for more details.
 **/

import events from 'node:events';
import { setTimeout } from 'node:timers/promises';

import { WebSocket } from 'ws';

import { Logger, getLogger } from 'common/log';
import { WsChannel } from './channel';

// need large message because resuming experiment will re-send all final metrics in one command
const maxPayload: number = 1024 * 1024 * 1024;

export class WsChannelClient extends WsChannel {
    private logger: Logger;  // avoid name conflict with base class
    private url: string;

    /**
     *  The url should start with "ws://".
     *  The name is used for better logging.
     **/
    constructor(url: string, name?: string) {
        const name_ = name ?? generateName(url);
        super(name_);
        this.logger = getLogger(`WsChannelClient.${name_}`);
        this.url = url;
        this.on('lost', this.reconnect.bind(this));
    }

    public async connect(): Promise<void> {
        this.logger.debug('Connecting to', this.url);
        const ws = new WebSocket(this.url, { maxPayload });
        this.setConnection(ws);
        await events.once(ws, 'open');
        this.logger.debug('Connected');
    }

    /**
     *  Alias of `close()`.
     **/
    public async disconnect(reason?: string): Promise<void> {
        this.close(reason ?? 'client disconnecting');
    }

    private async reconnect(): Promise<void> {
        this.logger.warning('Connection lost. Try to reconnect');
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

        this.logger.error('Conenction lost. Cannot reconnect');
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
