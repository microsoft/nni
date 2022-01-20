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
 *    5. Get rid of IOC.
 *    6. Deal with log path mismatch between REST API and file system.
 *    7. Strip slashes of URL prefix inside ExperimentStartupInfo.
 **/

import type { Server } from 'http';
import type { AddressInfo } from 'net';
import path from 'path';

import express, { Request, Response, Router } from 'express';
import httpProxy from 'http-proxy';
import { Deferred } from 'ts-deferred';

import { Singleton } from 'common/component';
import { getBasePort, getPrefixUrl } from 'common/experimentStartupInfo';
import { Logger, getLogger } from 'common/log';
import { getLogDir } from 'common/utils';
import { createRestHandler } from './restHandler';

/**
 *  The singleton REST server that dispatches web UI and `Experiment` requests.
 *
 *  RestServer must be initialized with start() after NNI manager constructing, but not necessarily after initializing.
 *  This is because RestServer needs NNI manager instance to register API handlers.
 **/
@Singleton
export class RestServer {
    private port: number;
    private urlPrefix: string;
    private server: Server | null = null;
    private logger: Logger = getLogger('RestServer');

    // I would prefer to get port and urlPrefix by constructor parameters,
    // but this is impossible due to limitation of IOC.
    constructor() {
        this.port = getBasePort();
        // Stripping slashes should be done inside ExperimentInfo, but I don't want to touch it for now.
        this.urlPrefix = '/' + stripSlashes(getPrefixUrl());
    }

    // The promise is resolved when it's ready to serve requests.
    // This worth nothing for now,
    // but for example if we connect to tuner using WebSocket then it must be launched after promise resolved.
    public start(): Promise<void> {
        this.logger.info(`Starting REST server at port ${this.port}, URL prefix: "${this.urlPrefix}"`);

        const app = express();
        // FIXME: We should have a global handler for critical errors.
        // `shutdown()` is not a callback and should not be passed to NNIRestHandler.
        app.use(this.urlPrefix, rootRouter(this.shutdown.bind(this)));
        app.all('*', (req, _res, next) => { console.log(req.url); next(); });
        app.all('*', (_req: Request, res: Response) => { res.status(404).send(`Outside prefix "${this.urlPrefix}"`); });
        console.log('### E ###');
        console.log(this.port);
        this.server = app.listen(this.port);

        const deferred = new Deferred<void>();
        this.server.on('listening', () => {
            console.log('### G ###');
            if (this.port === 0) {  // Currently for unit test, can be public feature in future.
                this.port = (<AddressInfo>this.server!.address()).port;
            }
            this.logger.info('REST server started.');
            console.log('### H ###');
            deferred.resolve();
            console.log('### I ###');
        });
        // FIXME: Use global handler. The event can be emitted after listening.
        this.server.on('error', (error: Error) => {
            this.logger.error('REST server error:', error);
            deferred.reject(error);
        });
        console.log('### F ###');
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
    router.use('/api/v1/nni', createRestHandler(stopCallback));

    /* Download log files */
    // The REST API path "/logs" does not match file system path "/log".
    // Here we use an additional router to workaround this problem.
    const logRouter = Router();
    logRouter.get('*', express.static(getLogDir()));
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

function stripSlashes(str: string): string {
    return str.replace(/^\/+/, '').replace(/\/+$/, '');
}

let webuiPath: string = path.resolve('static');
let netronUrl: string = 'https://netron.app';

export namespace UnitTestHelpers {
    export function getPort(server: RestServer): number {
        return (<any>server).port;
    }

    export function setWebuiPath(mockPath: string): void {
        webuiPath = path.resolve(mockPath);
    }

    export function setNetronUrl(mockUrl: string): void {
        netronUrl = mockUrl;
    }
}
