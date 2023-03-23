// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Currently the REST server that dispatches web UI and `Experiment` requests.
 *  In future it should handle WebSocket connections as well.
 *
 *  To add new APIs to REST server, modify `rootRouter()` function.
 *
 *  This file contains API URL constants. They must be synchronized with:
 *    - nni/experiment/rest.py
 *    - ts/webui/src/static/const.ts
 *    - ts/webui/src/components/public-child/OpenRow.tsx
 *  Remember to update them if the values are changed, or if this file is moved.
 *
 *  TODO:
 *    1. Refactor ClusterJobRestServer to an express-ws application so it doesn't require extra port.
 *    2. Provide public API to register express app, so this can be decoupled with other modules' implementation.
 *    3. Refactor NNIRestHandler. It's a mess.
 *    4. Deal with log path mismatch between REST API and file system.
 **/

import assert from 'assert/strict';
import type { Server } from 'http';
import type { AddressInfo } from 'net';
import path from 'path';

import express, { Request, Response, Router } from 'express';
import httpProxy from 'http-proxy';

import { Deferred } from 'common/deferred';
import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';

const logger: Logger = getLogger('RestServer');

/**
 *  The singleton REST server that dispatches web UI and `Experiment` requests.
 *
 *  RestServer must be initialized with start() after NNI manager constructing, but not necessarily after initializing.
 *  This is because RestServer needs NNI manager instance to register API handlers.
 **/
export class RestServer {
    private port: number;
    private urlPrefix: string;
    private server: Server | null = null;

    constructor(port: number, urlPrefix: string) {
        assert(!urlPrefix.startsWith('/') && !urlPrefix.endsWith('/'));
        this.port = port;
        this.urlPrefix = urlPrefix;
        globals.shutdown.register('RestServer', this.shutdown.bind(this));
    }

    // The promise is resolved when it's ready to serve requests.
    // This worth nothing for now,
    // but for example if we connect to tuner using WebSocket then it must be launched after promise resolved.
    public start(): Promise<void> {
        logger.info(`Starting REST server at port ${this.port}, URL prefix: "/${this.urlPrefix}"`);

        const app = globals.rest.getExpressApp();
        app.use('/' + this.urlPrefix, mainRouter());
        app.use('/' + this.urlPrefix, fallbackRouter());
        app.all('*', (_req: Request, res: Response) => { res.status(404).send(`Outside prefix "/${this.urlPrefix}"`); });
        this.server = app.listen(this.port);

        const deferred = new Deferred<void>();
        this.server.on('listening', () => {
            if (this.port === 0) {  // Currently for unit test, can be public feature in future.
                this.port = (<AddressInfo>this.server!.address()).port;
            }
            logger.info('REST server started.');
            deferred.resolve();
        });
        this.server.on('error', (error: Error) => { globals.shutdown.criticalError('RestServer', error); });
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
        setTimeout(() => {
            if (!deferred.settled) {
                logger.debug('Killing connections');
                this.server?.closeAllConnections();
            }
        }, 5000);
        return deferred.promise;
    }
}

/**
 *  You will need to modify this function if you want to add a new module, for example, project management.
 *
 *  Each module should have a unique URL prefix and a "Router". Check express' reference about Application and Router.
 *  Note that the order of `use()` calls does matter so you must not put a router after web UI.
 *  
 *  In fact experiments management should have a separate prefix and module.
 **/
function mainRouter(): Router {
    const router = globals.rest.getExpressRouter();

    /* Download log files */
    // The REST API path "/logs" does not match file system path "/log".
    // Here we use an additional router to workaround this problem.
    const logRouter = Router();
    logRouter.get('*', express.static(globals.paths.logDirectory));
    router.use('/logs', logRouter);

    /* NAS model visualization */
    router.use('/netron', netronProxy());
    return router;
}

function fallbackRouter(): Router {
    const router = Router();

    // The main router will handle this request once initialized.
    // So if it reaches here, the status is initializing.
    router.get('/api/v1/nni/check-status', (_req, res) => { res.send('INITIALIZING'); });

    /* Web UI */
    router.get('*', express.static(webuiPath));
    // React Router handles routing inside the browser. We must send index.html to all routes.
    // path.resolve() is required by Response.sendFile() API.
    router.get('*', (_req: Request, res: Response) => { res.sendFile(path.join(webuiPath, 'index.html')); });

    /* 404 as catch-all */
    router.all('*', (_req: Request, res: Response) => { res.status(404).send('Not Found'); });

    // TODO: websocket 404

    return router;
}

function netronProxy(): Router {
    const router = Router();
    const proxy = httpProxy.createProxyServer();
    router.all('*', (req: Request, res: Response): void => {
        delete req.headers.host;
        proxy.web(req, res, { changeOrigin: true, target: netronUrl });
    });
    return router;
}

let webuiPath: string = path.resolve('static');
let netronUrl: string = 'https://netron.app';

export namespace UnitTestHelpers {
    export function getPort(server: RestServer): number {
        return (server as any).port;
    }

    export function setWebuiPath(mockPath: string): void {
        webuiPath = path.resolve(mockPath);
    }

    export function setNetronUrl(mockUrl: string): void {
        netronUrl = mockUrl;
    }

    export function reset(): void {
        webuiPath = path.resolve('static');
        netronUrl = 'https://netron.app';
    }
}
