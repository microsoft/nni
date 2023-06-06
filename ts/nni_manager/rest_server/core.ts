// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  RestServerCore, a rest server which has no built-in handlers.
 *  All handlers will be registered with `globals.rest` APIs.
 *
 *  Currently it's a copy-paste of the main RestServer.
 *  In future RestServer will inherit this class.
 **/

import assert from 'node:assert/strict';
import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';

import { Deferred } from 'common/deferred';
import { globals } from 'common/globals';
import { Logger, getLogger } from 'common/log';

const logger: Logger = getLogger('RestServerCore');

export class RestServerCore {
    private port: number;
    private urlPrefix: string;
    private server: Server | null = null;

    /**
     *  When `port` is 0 or not given, let the OS to choose a random free port.
     **/
    constructor(port?: number, urlPrefix?: string) {
        this.port = port ?? 0;
        this.urlPrefix = urlPrefix ?? '';
        assert(!this.urlPrefix.startsWith('/') && !this.urlPrefix.endsWith('/'));
        globals.shutdown.register('RestServerCore', this.shutdown.bind(this));
    }

    public start(): Promise<void> {
        logger.info(`Starting REST server at port ${this.port}, URL prefix: "/${this.urlPrefix}"`);

        const app = globals.rest.getExpressApp();
        app.use('/' + this.urlPrefix, globals.rest.getExpressRouter());
        app.all('/' + this.urlPrefix, (_req, res) => { res.status(404).send('Not Found'); });
        app.all('*', (_req, res) => { res.status(404).send(`Outside prefix "/${this.urlPrefix}"`); });
        this.server = app.listen(this.port);

        const deferred = new Deferred<void>();
        this.server.on('listening', () => {
            if (this.port === 0) {
                this.port = (this.server!.address() as AddressInfo).port;
                (globals.args.port as any) = this.port;  // TODO: hacky, use globals.rest.port in future
            }
            logger.info('REST server started.');
            deferred.resolve();
        });
        this.server.on('error', error => { globals.shutdown.criticalError('RestServer', error); });
        return deferred.promise;
    }

    public shutdown(timeoutMilliseconds?: number): Promise<void> {
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
        setTimeout(() => {
            if (!deferred.settled) {
                logger.debug('Killing connections');
                this.server?.closeAllConnections();
            }
        }, timeoutMilliseconds ?? 5000);
        return deferred.promise;
    }
}
