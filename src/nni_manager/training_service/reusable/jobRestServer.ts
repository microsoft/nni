// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from 'events';
import { Request, Response, Router } from 'express';
import { ClusterJobRestServer } from '../common/clusterJobRestServer';
import { Server } from 'http';

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

    public get Server(): Server {
        return this.server;
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
