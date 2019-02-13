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
import * as path from 'path';

import * as component from '../common/component';
import { DataStore, MetricDataRecord, TrialJobInfo } from '../common/datastore';
import { NNIError, NNIErrorNames } from '../common/errors';
import { isNewExperiment } from '../common/experimentStartupInfo';
import { getLogger, Logger } from '../common/log';
import { ExperimentProfile, Manager, TrialJobStatistics} from '../common/manager';
import { ValidationSchemas } from './restValidationSchemas';
import { NNIRestServer } from './nniRestServer';

const expressJoi = require('express-joi-validator');

class NNIRestHandler {
    private restServer: NNIRestServer;
    private nniManager: Manager;
    private log: Logger;

    constructor(rs: NNIRestServer) {
        this.nniManager = component.get(Manager);
        this.restServer = rs;
        this.log = getLogger();
    }

    public createRestHandler(): Router {
        const router: Router = Router();

        // tslint:disable-next-line:typedef
        router.use((req: Request, res: Response, next) => {
            this.log.debug(`${req.method}: ${req.url}: body:\n${JSON.stringify(req.body, undefined, 4)}`);
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
            res.header('Access-Control-Allow-Methods', 'PUT,POST,GET,DELETE,OPTIONS');

            res.setHeader('Content-Type', 'application/json');
            next();
        });

        this.version(router);
        this.checkStatus(router);
        this.getExperimentProfile(router);
        this.updateExperimentProfile(router);
        this.startExperiment(router);
        this.getTrialJobStatistics(router);
        this.setClusterMetaData(router);
        this.listTrialJobs(router);
        this.getTrialJob(router);
        this.addTrialJob(router);
        this.cancelTrialJob(router);
        this.getMetricData(router);

        // Express-joi-validator configuration
        router.use((err: any, req: Request, res: Response, next: any) => {
            if (err.isBoom) {
                this.log.error(err.output.payload);

                return res.status(err.output.statusCode).json(err.output.payload);
            }
        });

        return router;
    }

    private handle_error(err: Error, res: Response, isFatal: boolean = false): void {
        if (err instanceof NNIError && err.name === NNIErrorNames.NOT_FOUND) {
            res.status(404);
        } else {
            res.status(500);
        }
        res.send({
            error: err.message
        });

        // If it's a fatal error, exit process
        if (isFatal) {
            this.log.fatal(err);
            process.exit(1);
        } else {
            this.log.error(err);
        }
    }

    private version(router: Router): void {
        router.get('/version', async (req: Request, res: Response) => {
            const pkg = await import(path.join(__dirname, '..', 'package.json'));
            res.send(pkg.version);
        });
    }

    // TODO add validators for request params, query, body
    private checkStatus(router: Router): void {
        router.get('/check-status', (req: Request, res: Response) => {
            const ds: DataStore = component.get<DataStore>(DataStore);
            ds.init().then(() => {
                res.send(this.nniManager.getStatus());
            }).catch(async (err: Error) => {
                this.handle_error(err, res);
                this.log.error(err.message);
                this.log.error(`Datastore initialize failed, stopping rest server...`);
                await this.restServer.stop();
            });
        });
    }

    private getExperimentProfile(router: Router): void {
        router.get('/experiment', (req: Request, res: Response) => {
            this.nniManager.getExperimentProfile().then((profile: ExperimentProfile) => {
                res.send(profile);
            }).catch((err: Error) => {
                this.handle_error(err, res);
            });
        });
    }

    private updateExperimentProfile(router: Router): void {
        router.put('/experiment', expressJoi(ValidationSchemas.UPDATEEXPERIMENT), (req: Request, res: Response) => {
            this.nniManager.updateExperimentProfile(req.body, req.query.update_type).then(() => {
                res.send();
            }).catch((err: Error) => {
                this.handle_error(err, res);
            });
        });
    }

    private startExperiment(router: Router): void {
        router.post('/experiment', expressJoi(ValidationSchemas.STARTEXPERIMENT), (req: Request, res: Response) => {
            if (isNewExperiment()) {
                this.nniManager.startExperiment(req.body).then((eid: string) => {
                    res.send({
                        experiment_id: eid
                    });
                }).catch((err: Error) => {
                    // Start experiment is a step of initialization, so any exception thrown is a fatal
                    this.handle_error(err, res);
                });
            } else {
                this.nniManager.resumeExperiment().then(() => {
                    res.send();
                }).catch((err: Error) => {
                    // Resume experiment is a step of initialization, so any exception thrown is a fatal
                    this.handle_error(err, res);
                });
            }
        });
    }

    private getTrialJobStatistics(router: Router): void {
        router.get('/job-statistics', (req: Request, res: Response) => {
            this.nniManager.getTrialJobStatistics().then((statistics: TrialJobStatistics[]) => {
                res.send(statistics);
            }).catch((err: Error) => {
                this.handle_error(err, res);
            });
        });
    }

    private setClusterMetaData(router: Router): void {
        router.put(
            '/experiment/cluster-metadata', expressJoi(ValidationSchemas.SETCLUSTERMETADATA),
            async (req: Request, res: Response) => {
            // tslint:disable-next-line:no-any
            const metadata: any = req.body;
            const keys: string[] = Object.keys(metadata);
            try {
                for (const key of keys) {
                    await this.nniManager.setClusterMetadata(key, JSON.stringify(metadata[key]));
                }
                res.send();
            } catch (err) {
                // setClusterMetata is a step of initialization, so any exception thrown is a fatal
                this.handle_error(err, res, true);
            }
        });
    }

    private listTrialJobs(router: Router): void {
        router.get('/trial-jobs', (req: Request, res: Response) => {
            this.nniManager.listTrialJobs(req.query.status).then((jobInfos: TrialJobInfo[]) => {
                jobInfos.forEach((trialJob: TrialJobInfo) => {
                    this.setErrorPathForFailedJob(trialJob);
                });
                res.send(jobInfos);
            }).catch((err: Error) => {
                this.handle_error(err, res);
            });
        });
    }

    private getTrialJob(router: Router): void {
        router.get('/trial-jobs/:id', (req: Request, res: Response) => {
            this.nniManager.getTrialJob(req.params.id).then((jobDetail: TrialJobInfo) => {
                const jobInfo: TrialJobInfo = this.setErrorPathForFailedJob(jobDetail);
                res.send(jobInfo);
            }).catch((err: Error) => {
                this.handle_error(err, res);
            });
        });
    }

    private addTrialJob(router: Router): void {
        router.post('/trial-jobs', async (req: Request, res: Response) => {
            this.nniManager.addCustomizedTrialJob(JSON.stringify(req.body)).then(() => {
                res.send();
            }).catch((err: Error) => {
                this.handle_error(err, res);
            });
        });
    }

    private cancelTrialJob(router: Router): void {
        router.delete('/trial-jobs/:id', async (req: Request, res: Response) => {
            this.nniManager.cancelTrialJobByUser(req.params.id).then(() => {
                res.send();
            }).catch((err: Error) => {
                this.handle_error(err, res);
            });
        });
    }

    private getMetricData(router: Router): void {
        router.get('/metric-data/:job_id*?', async (req: Request, res: Response) => {
            this.nniManager.getMetricData(req.params.job_id, req.query.type).then((metricsData: MetricDataRecord[]) => {
                res.send(metricsData);
            }).catch((err: Error) => {
                this.handle_error(err, res);
            });
        });
    }

    private setErrorPathForFailedJob(jobInfo: TrialJobInfo): TrialJobInfo {
        if (jobInfo === undefined || jobInfo.status !== 'FAILED' || jobInfo.logPath === undefined) {
            return jobInfo;
        }
        jobInfo.stderrPath = path.join(jobInfo.logPath, 'stderr');

        return jobInfo;
    }
}

export function createRestHandler(rs: NNIRestServer): Router {
    const handler: NNIRestHandler = new NNIRestHandler(rs);

    return handler.createRestHandler();
}
