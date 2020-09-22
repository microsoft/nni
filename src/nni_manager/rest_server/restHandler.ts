// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Request, Response, Router } from 'express';
import * as path from 'path';

import * as component from '../common/component';
import { DataStore, MetricDataRecord, TrialJobInfo } from '../common/datastore';
import { NNIError, NNIErrorNames } from '../common/errors';
import { isNewExperiment, isReadonly } from '../common/experimentStartupInfo';
import { getLogger, Logger } from '../common/log';
import { ExperimentProfile, Manager, TrialJobStatistics } from '../common/manager';
import { ValidationSchemas } from './restValidationSchemas';
import { NNIRestServer } from './nniRestServer';
import { getVersion } from '../common/utils';

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
        this.importData(router);
        this.getImportedData(router);
        this.startExperiment(router);
        this.getTrialJobStatistics(router);
        this.setClusterMetaData(router);
        this.listTrialJobs(router);
        this.getTrialJob(router);
        this.addTrialJob(router);
        this.cancelTrialJob(router);
        this.getMetricData(router);
        this.getMetricDataByRange(router);
        this.getLatestMetricData(router);
        this.getTrialLog(router);
        this.exportData(router);

        // Express-joi-validator configuration
        router.use((err: any, _req: Request, res: Response, _next: any) => {
            if (err.isBoom) {
                this.log.error(err.output.payload);

                return res.status(err.output.statusCode).json(err.output.payload);
            }
        });

        return router;
    }

    private handleError(err: Error, res: Response, isFatal: boolean = false, errorCode: number = 500): void {
        if (err instanceof NNIError && err.name === NNIErrorNames.NOT_FOUND) {
            res.status(404);
        } else {
            res.status(errorCode);
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
            const version = await getVersion();
            res.send(version);
        });
    }

    // TODO add validators for request params, query, body
    private checkStatus(router: Router): void {
        router.get('/check-status', (req: Request, res: Response) => {
            const ds: DataStore = component.get<DataStore>(DataStore);
            ds.init().then(() => {
                res.send(this.nniManager.getStatus());
            }).catch(async (err: Error) => {
                this.handleError(err, res);
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
                this.handleError(err, res);
            });
        });
    }

    private updateExperimentProfile(router: Router): void {
        router.put('/experiment', expressJoi(ValidationSchemas.UPDATEEXPERIMENT), (req: Request, res: Response) => {
            this.nniManager.updateExperimentProfile(req.body, req.query.update_type).then(() => {
                res.send();
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private importData(router: Router): void {
        router.post('/experiment/import-data', (req: Request, res: Response) => {
            this.nniManager.importData(JSON.stringify(req.body)).then(() => {
                res.send();
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getImportedData(router: Router): void {
        router.get('/experiment/imported-data', (req: Request, res: Response) => {
            this.nniManager.getImportedData().then((importedData: string[]) => {
                res.send(JSON.stringify(importedData));
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private startExperiment(router: Router): void {
        router.post('/experiment', expressJoi(ValidationSchemas.STARTEXPERIMENT), (req: Request, res: Response) => {
            if (isNewExperiment()) {
                this.nniManager.startExperiment(req.body).then((eid: string) => {
                    res.send({
                        experiment_id: eid // eslint-disable-line @typescript-eslint/camelcase
                    });
                }).catch((err: Error) => {
                    // Start experiment is a step of initialization, so any exception thrown is a fatal
                    this.handleError(err, res);
                });
            } else {
                this.nniManager.resumeExperiment(isReadonly()).then(() => {
                    res.send();
                }).catch((err: Error) => {
                    // Resume experiment is a step of initialization, so any exception thrown is a fatal
                    this.handleError(err, res);
                });
            } 
        });
    }

    private getTrialJobStatistics(router: Router): void {
        router.get('/job-statistics', (req: Request, res: Response) => {
            this.nniManager.getTrialJobStatistics().then((statistics: TrialJobStatistics[]) => {
                res.send(statistics);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private setClusterMetaData(router: Router): void {
        router.put(
            '/experiment/cluster-metadata', expressJoi(ValidationSchemas.SETCLUSTERMETADATA),
            async (req: Request, res: Response) => {
                const metadata: any = req.body;
                const keys: string[] = Object.keys(metadata);
                try {
                    for (const key of keys) {
                        await this.nniManager.setClusterMetadata(key, JSON.stringify(metadata[key]));
                    }
                    res.send();
                } catch (err) {
                    // setClusterMetata is a step of initialization, so any exception thrown is a fatal
                    this.handleError(NNIError.FromError(err), res, true);
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
                this.handleError(err, res);
            });
        });
    }

    private getTrialJob(router: Router): void {
        router.get('/trial-jobs/:id', (req: Request, res: Response) => {
            this.nniManager.getTrialJob(req.params.id).then((jobDetail: TrialJobInfo) => {
                const jobInfo: TrialJobInfo = this.setErrorPathForFailedJob(jobDetail);
                res.send(jobInfo);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private addTrialJob(router: Router): void {
        router.post('/trial-jobs', async (req: Request, res: Response) => {
            this.nniManager.addCustomizedTrialJob(JSON.stringify(req.body)).then((sequenceId: number) => {
                res.send({sequenceId});
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private cancelTrialJob(router: Router): void {
        router.delete('/trial-jobs/:id', async (req: Request, res: Response) => {
            this.nniManager.cancelTrialJobByUser(req.params.id).then(() => {
                res.send();
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getMetricData(router: Router): void {
        router.get('/metric-data/:job_id*?', async (req: Request, res: Response) => {
            this.nniManager.getMetricData(req.params.job_id, req.query.type).then((metricsData: MetricDataRecord[]) => {
                res.send(metricsData);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getMetricDataByRange(router: Router): void {
        router.get('/metric-data-range/:min_seq_id/:max_seq_id', async (req: Request, res: Response) => {
            const minSeqId = Number(req.params.min_seq_id);
            const maxSeqId = Number(req.params.max_seq_id);
            this.nniManager.getMetricDataByRange(minSeqId, maxSeqId).then((metricsData: MetricDataRecord[]) => {
                res.send(metricsData);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getLatestMetricData(router: Router): void {
        router.get('/metric-data-latest/', async (req: Request, res: Response) => {
            this.nniManager.getLatestMetricData().then((metricsData: MetricDataRecord[]) => {
                res.send(metricsData);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getTrialLog(router: Router): void {
        router.get('/trial-log/:id/:type', async(req: Request, res: Response) => {
            this.nniManager.getTrialLog(req.params.id, req.params.type).then((log: string) => {
                if (log === '') {
                    log = 'No logs available.'
                }
                res.send(log);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private exportData(router: Router): void {
        router.get('/export-data', (req: Request, res: Response) => {
            this.nniManager.exportData().then((exportedData: string) => {
                res.send(exportedData);
            }).catch((err: Error) => {
                this.handleError(err, res);
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
