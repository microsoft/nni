// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as bodyParser from 'body-parser';
import * as express from 'express';
import * as path from 'path';
import * as component from '../common/component';
import { RestServer } from '../common/restServer'
import { getLogDir } from '../common/utils';
import { createRestHandler } from './restHandler';

/**
 * NNI Main rest server, provides rest API to support
 * # nnictl CLI tool
 * # NNI WebUI
 *
 */
@component.Singleton
export class NNIRestServer extends RestServer {
    private readonly API_ROOT_URL: string = '/api/v1/nni';
    private readonly LOGS_ROOT_URL: string = '/logs';

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super();
    }

    /**
     * NNIRestServer's own router registration
     */
    protected registerRestHandler(): void {
        this.app.use(express.static('static'));
        this.app.use(bodyParser.json({limit: '50mb'}));
        this.app.use(this.API_ROOT_URL, createRestHandler(this));
        this.app.use(this.LOGS_ROOT_URL, express.static(getLogDir()));
        this.app.get('*', (req: express.Request, res: express.Response) => {
            res.sendFile(path.resolve('static/index.html'));
        });
    }
}
