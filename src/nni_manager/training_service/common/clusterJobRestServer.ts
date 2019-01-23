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
import { Request, Response, Router } from 'express';
import * as bodyParser from 'body-parser';
import * as component from '../../common/component';
import * as fs from 'fs'
import * as path from 'path'
import { getBasePort, getExperimentId } from '../../common/experimentStartupInfo';
import { RestServer } from '../../common/restServer'
import { getLogDir } from '../../common/utils';
import { Writable } from 'stream';

/**
 * Cluster Job Training service Rest server, provides rest API to support Cluster job metrics update
 * 
 */
@component.Singleton
export abstract class ClusterJobRestServer extends RestServer{
    private readonly API_ROOT_URL: string = '/api/v1/nni-pai';
    private readonly NNI_METRICS_PATTERN: string = `NNISDK_MEb'(?<metrics>.*?)'`;

    private readonly expId: string = getExperimentId();

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super();
        const basePort: number = getBasePort();
        assert(basePort && basePort > 1024);
        
        this.port = basePort + 1;         
    }

    public get clusterRestServerPort(): number {
        if(!this.port) {
            throw new Error('PAI Rest server port is undefined');
        }
        return this.port;
    }

    /**
     * NNIRestServer's own router registration
     */
    protected registerRestHandler(): void {
        this.app.use(bodyParser.json());
        this.app.use(this.API_ROOT_URL, this.createRestHandler());
    }

    private createRestHandler() : Router {
        const router: Router = Router();

        // tslint:disable-next-line:typedef
        router.use((req: Request, res: Response, next) => {
            this.log.info(`${req.method}: ${req.url}: body:\n${JSON.stringify(req.body, undefined, 4)}`);
            res.setHeader('Content-Type', 'application/json');
            next();
        });

        router.post(`/update-metrics/${this.expId}/:trialId`, (req: Request, res: Response) => {
            try {
                this.log.info(`Get update-metrics request, trial job id is ${req.params.trialId}`);
                this.log.info(`update-metrics body is ${JSON.stringify(req.body)}`);

                this.handleTrialMetrics(req.body.jobId, req.body.metrics);

                res.send();
            }
            catch(err) {
                this.log.error(`json parse metrics error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });

        router.post(`/stdout/${this.expId}/:trialId`, (req: Request, res: Response) => {
            const trialLogPath: string = path.join(getLogDir(), `trial_${req.params.trialId}.log`);
            try {
                let skipLogging: boolean = false;
                if(req.body.tag === 'trial' && req.body.msg !== undefined) {
                    const metricsContent = req.body.msg.match(this.NNI_METRICS_PATTERN);
                    if(metricsContent && metricsContent.groups) {
                        this.handleTrialMetrics(req.params.trialId, [metricsContent.groups['metrics']]);
                        skipLogging = true;
                    }
                }

                if(!skipLogging){
                    // Construct write stream to write remote trial's log into local file
                    const writeStream: Writable = fs.createWriteStream(trialLogPath, {
                        flags: 'a+',
                        encoding: 'utf8',
                        autoClose: true
                    });

                    writeStream.write(req.body.msg + '\n');
                    writeStream.end();
                }
                res.send();
            }
            catch(err) {
                this.log.error(`json parse stdout data error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });

        return router;
    }

    /** Abstract method to handle trial metrics data */
    protected abstract handleTrialMetrics(jobId : string, trialMetrics : any[]) : void;
}