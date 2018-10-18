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

import { Request, Response, Router } from 'express';
import * as bodyParser from 'body-parser';
import * as component from '../../common/component';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { Inject } from 'typescript-ioc';
import { PAITrainingService } from './paiTrainingService';
import { RestServer } from '../../common/restServer'

/**
 * PAI Training service Rest server, provides rest API to support pai job metrics update
 * 
 */
@component.Singleton
export class PAIJobRestServer extends RestServer{
    /** NNI main rest service default port */
    private static readonly DEFAULT_PORT: number = 51189;

    private readonly API_ROOT_URL: string = '/api/v1/nni-pai';

    private readonly expId: string = getExperimentId();

    @Inject
    private readonly paiTrainingService : PAITrainingService;

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super();
        this.port = PAIJobRestServer.DEFAULT_PORT;
        this.paiTrainingService = component.get(PAITrainingService);
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

                // Split metrics array into single metric, then emit
                // Warning: If not split metrics into single ones, the behavior will be UNKNOWN
                for (const singleMetric of req.body.metrics) {
                    this.paiTrainingService.MetricsEmitter.emit('metric', {
                        id : req.body.jobId,
                        data : singleMetric
                    });
                }

                res.send();
            }
            catch(err) {
                this.log.error(`json parse metrics error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });

        router.get(`/task/${this.expId}/trialId`, (req: Request, res: Response) => {
            try {
                
                this.log.info(`Get task request, trial job id is ${req.params.trialId}`);
                if(this.paiTrainingService.taskQueue.length > 1){
                    console.log('--------------get task request----------')
                    this.paiTrainingService.taskQueue.shift();
                    res.send({"task": this.paiTrainingService.taskQueue[0]});
                }else{
                    res.send({"task": null});
                }  
            }
            catch(err) {
                res.status(500);
                res.send(err.message);
            }
        });

        router.get(`/report/${this.expId}/trialId`, (req: Request, res: Response) => {
            try {
                console.log('--------------get report request----------')
                console.log(req.params.trialId)
                this.paiTrainingService.copyDataFromHdfs(req.params.trialId).then(()=>{

                });
                res.send()
            }
            catch(err) {
                res.status(500);
                res.send(err.message);
            }
        });

        return router;
    }
}