// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import bodyParser from 'body-parser';
import express from 'express';
import httpProxy from 'http-proxy';
import path from 'path';
import * as component from '../common/component';
import { RestServer } from '../common/restServer'
import { getLogDir } from '../common/utils';
import { createRestHandler } from './restHandler';
import { getAPIRootUrl, getPrefixUrl } from '../common/experimentStartupInfo';

/**
 * NNI Main rest server, provides rest API to support
 * # nnictl CLI tool
 * # NNI WebUI
 *
 */
@component.Singleton
export class NNIRestServer extends RestServer {
    private readonly LOGS_ROOT_URL: string = '/logs';
    protected netronProxy: any = null;
    protected API_ROOT_URL: string = '/api/v1/nni';

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super();
        this.API_ROOT_URL = getAPIRootUrl();
        this.netronProxy = httpProxy.createProxyServer();
    }

    /**
     * NNIRestServer's own router registration
     */
    protected registerRestHandler(): void {
        this.app.use(getPrefixUrl(), express.static('static'));
        this.app.use(bodyParser.json({limit: '50mb'}));
        this.app.use(this.API_ROOT_URL, createRestHandler(this));
        this.app.use(this.LOGS_ROOT_URL, express.static(getLogDir()));
        this.app.all('/netron/*', (req: express.Request, res: express.Response) => {
            delete req.headers.host;
            req.url = req.url.replace('/netron', '/');
            this.netronProxy.web(req, res, {
                changeOrigin: true,
                target: 'https://netron.app'
            });
        });
        this.app.get(`${getPrefixUrl()}/*`, (_req: express.Request, res: express.Response) => {
            res.sendFile(path.resolve('static/index.html'));
        });
    }
}
