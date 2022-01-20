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
 *    - ts/webui/src/static/constant.ts
 *    - ts/webui/src/components/public-child/OpenRow.tsx
 *  Remember to update them if the values are changed, or if this file is moved.
 *
 *  TODO:
 *    1. Add a global function to handle critical error.
 *    2. Refactor ClusterJobRestServer to an express-ws application so it don't require extra port.
 *    3. Provide public API to register express app, so this can be decoupled with other modules' implementation.
 *    4. Refactor NNIRestHandler. It's a mess.
 *    5. Get rid of IOC.
 **/

import type { Server } from 'http';
import path from 'path';

import bodyParser from 'body-parser';
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
        this.urlPrefix = getPrefixUrl();
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
        app.all('*', (_req: Request, res: Response) => { res.status(404).send(`Outside prefix "${this.urlPrefix}"`); });
        this.server = app.listen(this.port);

        const deferred = new Deferred<void>();
        this.server.on('listening', () => {
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
    router.use(bodyParser.json({ limit: '50mb' }));

    /* NNI manager APIs */
    router.use('/api/v1/nni', createRestHandler(stopCallback));

    /* Download log files */
    router.use('/logs', express.static(getLogDir()));

    /* NAS model visualization */
    router.use('/netron', netronProxy());

    /* Web UI */
    router.use('/', express.static('static'));
    // React Router handles routing inside the browser. We must send index.html to all routes.
    // path.resolve() is required by Response.sendFile() API.
    router.get('*', (_req: Request, res: Response) => { res.sendFile(path.resolve('static/index.html')); });

    /* 404 as catch-all */
    router.all('*', (_req: Request, res: Response) => { res.status(404).send('Not Found'); });
    return router;
}

function netronProxy(): Router {
    const router = Router();
    const proxy = httpProxy.createProxyServer();
    router.all('*', (req: Request, res: Response): void => {
        delete req.headers.host;
        proxy.web(req, res, { changeOrigin: true, target: 'https://netron.app' });
    });
    return router;
}
