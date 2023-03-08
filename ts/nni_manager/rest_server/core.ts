// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'node:assert/strict';
import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';

import express from 'express';
import expressWs from 'express-ws';

import { Deferred } from 'common/deferred';
import { globals } from 'common/globals';
import { Logger, getLogger } from 'common/log';

const logger: Logger = getLogger('RestServerCore');

export class RestServerCore {
    private port: number;
    private urlPrefix: string;
    private server: Server | null = null;

    constructor(port?: number, urlPrefix?: string) {
        this.port = port ?? 0;
        this.urlPrefix = urlPrefix ?? '';
        assert(!this.urlPrefix.startsWith('/') && !this.urlPrefix.endsWith('/'));
        globals.shutdown.register('RestServerCore', this.shutdown.bind(this));
    }

    public start(): Promise<void> {
        logger.info(`Starting REST server at port ${this.port}, URL prefix: "/${this.urlPrefix}"`);

        const app = express();
        expressWs(app, undefined, { wsOptions: { maxPayload: 4 * 1024 * 1024 * 1024 }});

        app.use('/' + this.urlPrefix, globals.rest.getExpressRouter());
        app.all('/' + this.urlPrefix, (_req, res) => { res.status(404).send('Not Found'); });
        app.all('*', (_req, res) => { res.status(404).send(`Outside prefix "/${this.urlPrefix}"`); });
        this.server = app.listen(this.port);

        const deferred = new Deferred<void>();
        this.server.on('listening', () => {
            if (this.port === 0) {
                this.port = (this.server!.address() as AddressInfo).port;
            }
            logger.info('REST server started.');
            deferred.resolve();
        });
        this.server.on('error', error => { globals.shutdown.criticalError('RestServer', error); });
        return deferred.promise;
    }

    public shutdown(): Promise<void> {
        logger.info('Stopping REST server.');
        if (this.server === null) {
            logger.warning('REST server is not running.');
            return Promise.resolve();
        }
        const deferred = new Deferred<void>();
        this.server.close(() => {
            logger.info('REST server stopped.');
            deferred.resolve();
        });
        return deferred.promise;
    }
}