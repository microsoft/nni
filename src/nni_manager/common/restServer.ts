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

import * as assert from 'assert';
import * as express from 'express';
import * as http from 'http';
import { Deferred } from 'ts-deferred';
import { getLogger, Logger } from './log';
import { getBasePort } from './experimentStartupInfo';


/**
 * Abstraction class to create a RestServer
 * The module who wants to use a RestServer could <b>extends</b> this abstract class
 * And implement its own registerRestHandler() function to register routers
 */
export abstract class RestServer {
    private startTask!: Deferred<void>;
    private stopTask!: Deferred<void>;
    private server!: http.Server;

    /** The fields can be inherited by subclass */
    protected hostName: string = '0.0.0.0';
    protected port?: number;
    protected app: express.Application = express();
    protected log: Logger = getLogger();
    protected basePort?: number;

    constructor() {
        this.port = getBasePort();
        assert(this.port && this.port > 1024);
    }

    get endPoint(): string {
        // tslint:disable-next-line:no-http-string
        return `http://${this.hostName}:${this.port}`;
    }

    public start(hostName?: string): Promise<void> {
        this.log.info(`RestServer start`);
        if (this.startTask !== undefined) {
            return this.startTask.promise;
        }
        this.startTask = new Deferred<void>();

        this.registerRestHandler();

        if (hostName) {
            this.hostName = hostName;
        }

        this.log.info(`RestServer base port is ${this.port}`);

        this.server = this.app.listen(this.port as number, this.hostName).on('listening', () => {
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
                    //Stops the server from accepting new connections and keeps existing connections.
                    //This function is asynchronous, the server is finally closed when all connections
                    //are ended and the server emits a 'close' event.
                    //Refer https://nodejs.org/docs/latest/api/net.html#net_server_close_callback
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
        this.stopTask.resolve()
        return this.stopTask.promise;
    }

    /**
     * Register REST handler, which is left for subclass to implement
     */
    protected abstract registerRestHandler(): void;
}
