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

import { EventEmitter } from 'events';
import { Request, Response, Router } from 'express';
import { ClusterJobRestServer } from '../../common/clusterJobRestServer';

export interface ParameterFileMeta {
    readonly experimentId: string;
    readonly trialId: string;
    readonly filePath: string;
}

/**
 * TODO: it should be merged into ClusterJobRestServer
 */
export class JobRestServer extends ClusterJobRestServer {
    protected parameterFileMetaList: ParameterFileMeta[] = [];

    protected readonly metricsEmitter: EventEmitter;

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor(metricsEmitter: EventEmitter) {
        super();
        this.metricsEmitter = metricsEmitter;
        this.setEnableVersionCheck = true;
    }

    protected handleTrialMetrics(jobId: string, metrics: any[]): void {
        // Split metrics array into single metric, then emit
        // Warning: If not split metrics into single ones, the behavior will be UNKNOWN
        for (const singleMetric of metrics) {
            this.metricsEmitter.emit('metric', {
                id: jobId,
                data: singleMetric
            });
        }
    }

    protected createRestHandler(): Router {
        const router: Router = super.createRestHandler();

        router.post(`/parameter-file-meta`, (req: Request, res: Response) => {
            try {
                this.log.info(`POST /parameter-file-meta, body is ${JSON.stringify(req.body)}`);
                this.parameterFileMetaList.push(req.body);
                res.send();
            } catch (err) {
                this.log.error(`POST parameter-file-meta error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });

        router.get(`/parameter-file-meta`, (req: Request, res: Response) => {
            try {
                this.log.info(`GET /parameter-file-meta`);
                res.send(this.parameterFileMetaList);
            } catch (err) {
                this.log.error(`GET parameter-file-meta error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });

        return router;
    }
}
