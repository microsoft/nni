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
 *    1. Add a global function to handle critical error.
 *    2. Refactor ClusterJobRestServer to an express-ws application so it doesn't require extra port.
 *    3. Provide public API to register express app, so this can be decoupled with other modules' implementation.
 *    4. Refactor NNIRestHandler. It's a mess.
 *    5. Deal with log path mismatch between REST API and file system.
 **/

import assert from 'assert/strict';
import type { Server } from 'http';
import type { AddressInfo } from 'net';
import path from 'path';

import express, { Request, Response, Router } from 'express';
import httpProxy from 'http-proxy';
import { Deferred } from 'ts-deferred';

import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';
import { createRestHandler } from './restHandler';

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
    private logger: Logger = getLogger('RestServer');

    constructor(port: number, urlPrefix: string) {
        assert(!urlPrefix.startsWith('/') && !urlPrefix.endsWith('/'));
        this.port = port;
        this.urlPrefix = urlPrefix;
    }

    // The promise is resolved when it's ready to serve requests.
    // This worth nothing for now,
    // but for example if we connect to tuner using WebSocket then it must be launched after promise resolved.
    public start(): Promise<void> {
        this.logger.info(`Starting REST server at port ${this.port}, URL prefix: "/${this.urlPrefix}"`);

        const app = express();
        // FIXME: We should have a global handler for critical errors.
        // `shutdown()` is not a callback and should not be passed to NNIRestHandler.
        app.use('/' + this.urlPrefix, rootRouter(this.shutdown.bind(this)));
        app.all('*', (_req: Request, res: Response) => { res.status(404).send(`Outside prefix "/${this.urlPrefix}"`); });
        this.server = app.listen(this.port);

        const deferred = new Deferred<void>();
        this.server.on('listening', () => {
            if (this.port === 0) {  // Currently for unit test, can be public feature in future.
                this.port = (<AddressInfo>this.server!.address()).port;
            }
            this.logger.info('REST server started.');
            deferred.resolve();
        });
        // FIXME: Use global handler. The event can be emitted after listening.
        this.server.on('error', (error: Error) => {
            this.logger.error('REST server error:', error);
            deferred.reject(error);
        });
        return deferred.promise;
    }

    public shutdown(): Promise<void> {
        this.logger.info('Stopping REST server.');
        if (this.server === null) {
            this.logger.warning('REST server is not running.');
            return Promise.resolve();
        }
        const deferred = new Deferred<void>();
        this.server.close(() => {
            this.logger.info('REST server stopped.');
            deferred.resolve();
        });
        // FIXME: Use global handler. It should be aware of shutting down event and swallow errors in this stage.
        this.server.on('error', (error: Error) => {
            this.logger.error('REST server error:', error);
            deferred.resolve();
        });
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
function rootRouter(stopCallback: () => Promise<void>): Router {
    const router = Router();
    router.use(express.json({ limit: '50mb' }));

    /* NNI manager APIs */
    router.use('/api/v1/nni', restHandlerFactory(stopCallback));

    /* Download log files */
    // The REST API path "/logs" does not match file system path "/log".
    // Here we use an additional router to workaround this problem.
    const logRouter = Router();
    logRouter.get('*', express.static(globals.paths.logDirectory));
    router.use('/logs', logRouter);

    /* NAS model visualization */
    router.use('/netron', netronProxy());

    /* Web UI */
    router.get('*', express.static(webuiPath));
    // React Router handles routing inside the browser. We must send index.html to all routes.
    // path.resolve() is required by Response.sendFile() API.
    router.get('*', (_req: Request, res: Response) => { res.sendFile(path.join(webuiPath, 'index.html')); });

    /* 404 as catch-all */
    router.all('*', (_req: Request, res: Response) => { res.status(404).send('Not Found'); });
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
let restHandlerFactory = createRestHandler;

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

    export function disableNniManager(): void {
        restHandlerFactory = (_: any): Router => Router();
    }

    export function reset(): void {
        webuiPath = path.resolve('static');
        netronUrl = 'https://netron.app';
        restHandlerFactory = createRestHandler;
    }
}
