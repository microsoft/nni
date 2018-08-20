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
import * as http from 'http';
import { Deferred } from 'ts-deferred';

import * as component from '../common/component';
import { getLogger, Logger } from '../common/log';
import { Manager } from '../common/manager';
import { createRestHandler } from './restHandler';

@component.Singleton
export class RestServer {
    public static readonly DEFAULT_PORT: number = 51188;
    private readonly API_ROOT_URL: string = '/api/v1/nni';
    private hostName: string = '0.0.0.0';
    private port: number = RestServer.DEFAULT_PORT;
    private startTask!: Deferred<void>;
    private stopTask!: Deferred<void>;
    private app: express.Application = express();
    private server!: http.Server;
    private log: Logger = getLogger();

    get endPoint(): string {
        // tslint:disable-next-line:no-http-string
        return `http://${this.hostName}:${this.port}`;
    }

    public start(port?: number, hostName?: string): Promise<void> {
        if (this.startTask !== undefined) {
            return this.startTask.promise;
        }
        this.startTask = new Deferred<void>();

        this.registerRestHandler();

        if (hostName) {
            this.hostName = hostName;
        }
        if (port) {
            this.port = port;
        }

        this.server = this.app.listen(this.port, this.hostName).on('listening', () => {
            this.startTask.resolve();
        }).on('error', (e: Error) => {
            this.startTask.reject(e);
        });

        return this.startTask.promise;
    }

    public stop(): Promise<void> {
        if (this.stopTask !== undefined) {
            return this.stopTask.promise;
        }
        this.stopTask = new Deferred<void>();

        if (this.startTask === undefined) {
            this.stopTask.resolve();

            return this.stopTask.promise;
        } else {
            this.startTask.promise.then(
                () => { // Started
                    this.server.close().on('close', () => {
                        this.log.info('Rest server stopped.');
                        this.stopTask.resolve();
                    }).on('error', (e: Error) => {
                        this.log.error(`Error occurred stopping Rest server: ${e.message}`);
                        this.stopTask.reject();
                    });
                },
                () => { // Start task rejected
                    this.stopTask.resolve();
                }
            );
        }

        return this.stopTask.promise;
    }

    private registerRestHandler(): void {
        this.app.use(bodyParser.json());
        this.app.use(this.API_ROOT_URL, createRestHandler(this));
    }
}
