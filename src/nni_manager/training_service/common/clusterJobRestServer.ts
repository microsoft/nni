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
// tslint:disable-next-line:no-implicit-dependencies
import * as bodyParser from 'body-parser';
import { Request, Response, Router } from 'express';
import * as fs from 'fs';
import * as path from 'path';
import { Writable } from 'stream';
import { String } from 'typescript-string-operations';
import * as component from '../../common/component';
import { getBasePort, getExperimentId } from '../../common/experimentStartupInfo';
import { RestServer } from '../../common/restServer';
import { getLogDir } from '../../common/utils';

/**
 * Cluster Job Training service Rest server, provides rest API to support Cluster job metrics update
 *
 */
@component.Singleton
export abstract class ClusterJobRestServer extends RestServer {
    private readonly API_ROOT_URL: string = '/api/v1/nni-pai';
    private readonly NNI_METRICS_PATTERN: string = `NNISDK_MEb'(?<metrics>.*?)'`;

    private readonly expId: string = getExperimentId();

    private enableVersionCheck: boolean = true; //switch to enable version check
    private versionCheckSuccess: boolean | undefined;
    private errorMessage?: string;

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super();
        const basePort: number = getBasePort();
        assert(basePort !== undefined && basePort > 1024);

        this.port = basePort + 1;
    }

    get apiRootUrl(): string {
        return this.API_ROOT_URL;
    }

    public get clusterRestServerPort(): number {
        if (this.port === undefined) {
            throw new Error('PAI Rest server port is undefined');
        }

        return this.port;
    }

    public get getErrorMessage(): string | undefined {
        return this.errorMessage;
    }

    public set setEnableVersionCheck(versionCheck: boolean) {
        this.enableVersionCheck = versionCheck;
    }

    /**
     * NNIRestServer's own router registration
     */
    protected registerRestHandler(): void {
        this.app.use(bodyParser.json());
        this.app.use(this.API_ROOT_URL, this.createRestHandler());
    }

    // Abstract method to handle trial metrics data
    // tslint:disable-next-line:no-any
    protected abstract handleTrialMetrics(jobId : string, trialMetrics : any[]) : void;

    // tslint:disable: no-unsafe-any no-any
    protected createRestHandler() : Router {
        const router: Router = Router();

        router.use((req: Request, res: Response, next: any) => {
            this.log.info(`${req.method}: ${req.url}: body:\n${JSON.stringify(req.body, undefined, 4)}`);
            res.setHeader('Content-Type', 'application/json');
            next();
        });

        router.post(`/version/${this.expId}/:trialId`, (req: Request, res: Response) => {
            if (this.enableVersionCheck) {
                try {
                    const checkResultSuccess: boolean = req.body.tag === 'VCSuccess' ? true : false;
                    if (this.versionCheckSuccess !== undefined && this.versionCheckSuccess !== checkResultSuccess) {
                        this.errorMessage = 'Version check error, version check result is inconsistent!';
                        this.log.error(this.errorMessage);
                    } else if (checkResultSuccess) {
                        this.log.info(`Version check in trialKeeper success!`);
                        this.versionCheckSuccess = true;
                    } else {
                        this.versionCheckSuccess = false;
                        this.errorMessage = req.body.msg;
                    }
                } catch (err) {
                    this.log.error(`json parse metrics error: ${err}`);
                    res.status(500);
                    res.send(err.message);
                }
            } else {
                this.log.info(`Skipping version check!`);
            }
            res.send();
        });

        router.post(`/update-metrics/${this.expId}/:trialId`, (req: Request, res: Response) => {
            try {
                this.log.info(`Get update-metrics request, trial job id is ${req.params.trialId}`);
                this.log.info(`update-metrics body is ${JSON.stringify(req.body)}`);

                this.handleTrialMetrics(req.body.jobId, req.body.metrics);

                res.send();
            } catch (err) {
                this.log.error(`json parse metrics error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });

        router.post(`/stdout/${this.expId}/:trialId`, (req: Request, res: Response) => {
            if (this.enableVersionCheck && (this.versionCheckSuccess === undefined || !this.versionCheckSuccess)
            && this.errorMessage === undefined) {
                this.errorMessage = `Version check failed, didn't get version check response from trialKeeper,`
                 + ` please check your NNI version in NNIManager and TrialKeeper!`;
            }
            const trialLogPath: string = path.join(getLogDir(), `trial_${req.params.trialId}.log`);
            try {
                let skipLogging: boolean = false;
                if (req.body.tag === 'trial' && req.body.msg !== undefined) {
                    const metricsContent: any = req.body.msg.match(this.NNI_METRICS_PATTERN);
                    if (metricsContent && metricsContent.groups) {
                        const key: string = 'metrics';
                        this.handleTrialMetrics(req.params.trialId, [metricsContent.groups[key]]);
                        skipLogging = true;
                    }
                }

                if (!skipLogging) {
                    // Construct write stream to write remote trial's log into local file
                    // tslint:disable-next-line:non-literal-fs-path
                    const writeStream: Writable = fs.createWriteStream(trialLogPath, {
                        flags: 'a+',
                        encoding: 'utf8',
                        autoClose: true
                    });

                    writeStream.write(String.Format('{0}\n', req.body.msg));
                    writeStream.end();
                }
                res.send();
            } catch (err) {
                this.log.error(`json parse stdout data error: ${err}`);
                res.status(500);
                res.send(err.message);
            }
        });

        return router;
    }
    // tslint:enable: no-unsafe-any no-any
}
