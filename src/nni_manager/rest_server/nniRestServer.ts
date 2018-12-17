/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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
        this.app.use(bodyParser.json());
        this.app.use(this.API_ROOT_URL, createRestHandler(this));
        this.app.use(this.LOGS_ROOT_URL, express.static(getLogDir()));
        this.app.get('*', (req: express.Request, res: express.Response) => {
            res.sendFile(path.resolve('static/index.html'));
        });
    }
}
