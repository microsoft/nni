// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { Request, Response, Router } from 'express';
import path from 'path';

import { IocShim } from 'common/ioc_shim';
import { DataStore, MetricDataRecord, TrialJobInfo } from '../common/datastore';
import { NNIError, NNIErrorNames } from '../common/errors';
import { isNewExperiment, isReadonly } from '../common/experimentStartupInfo';
import globals from 'common/globals';
import { getLogger, Logger } from '../common/log';
import { ExperimentProfile, Manager, TrialJobStatistics } from '../common/manager';
import { getExperimentsManager } from 'extensions/experiments_manager';
import { TensorboardManager, TensorboardTaskInfo } from '../common/tensorboardManager';
import { getVersion } from '../common/utils';
import { MetricType } from '../common/datastore';
import { ProfileUpdateType } from '../common/manager';
import { TrialJobStatus } from '../common/trainingService';

// TODO: fix expressJoi
//const expressJoi = require('express-joi-validator');

class NNIRestHandler {
    private nniManager: Manager;
    private tensorboardManager: TensorboardManager;
    private log: Logger;

    constructor() {
        this.nniManager = IocShim.get(Manager);
        this.tensorboardManager = IocShim.get(TensorboardManager);
        this.log = getLogger('NNIRestHandler');
    }

    public createRestHandler(): Router {
        const router: Router = Router();

        router.use((req: Request, res: Response, next) => {
            this.log.debug(`${req.method}: ${req.url}: body:`, req.body);
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
            res.header('Access-Control-Allow-Methods', 'PUT,POST,GET,DELETE,OPTIONS');

            res.setHeader('Content-Type', 'application/json');
            next();
        });

        this.version(router);
        this.checkStatus(router);
        this.getExperimentProfile(router);
        this.getExperimentMetadata(router);
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
        this.getTrialFile(router);
        this.exportData(router);
        this.getExperimentsInfo(router);
        this.startTensorboardTask(router);
        this.getTensorboardTask(router);
        this.updateTensorboardTask(router);
        this.stopTensorboardTask(router);
        this.stopAllTensorboardTask(router);
        this.listTensorboardTask(router);
        this.stop(router);

        // Express-joi-validator configuration
        router.use((err: any, _req: Request, res: Response, _next: any): any => {
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
            this.log.critical(err);
            process.exit(1);
        } else {
            this.log.error(err);
        }
    }

    private version(router: Router): void {
        router.get('/version', async (_req: Request, res: Response) => {
            const version = await getVersion();
            res.send(version);
        });
    }

    // TODO add validators for request params, query, body
    private checkStatus(router: Router): void {
        router.get('/check-status', (_req: Request, res: Response) => {
            const ds: DataStore = IocShim.get<DataStore>(DataStore);
            ds.init().then(() => {
                res.send(this.nniManager.getStatus());
            }).catch(async (err: Error) => {
                this.handleError(err, res);
                this.log.error(err.message);
                this.log.error(`Datastore initialize failed, stopping rest server...`);
                globals.shutdown.criticalError('RestHandler', err);
            });
        });
    }

    private getExperimentProfile(router: Router): void {
        router.get('/experiment', (_req: Request, res: Response) => {
            this.nniManager.getExperimentProfile().then((profile: ExperimentProfile) => {
                res.send(profile);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private updateExperimentProfile(router: Router): void {
        router.put('/experiment', (req: Request, res: Response) => {
            this.nniManager.updateExperimentProfile(req.body, req.query['update_type'] as ProfileUpdateType).then(() => {
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
        router.get('/experiment/imported-data', (_req: Request, res: Response) => {
            this.nniManager.getImportedData().then((importedData: string[]) => {
                res.send(JSON.stringify(importedData));
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private startExperiment(router: Router): void {
        router.post('/experiment', (req: Request, res: Response) => {
            if (isNewExperiment()) {
                this.nniManager.startExperiment(req.body).then((eid: string) => {
                    res.send({
                        experiment_id: eid
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
        router.get('/job-statistics', (_req: Request, res: Response) => {
            this.nniManager.getTrialJobStatistics().then((statistics: TrialJobStatistics[]) => {
                res.send(statistics);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private setClusterMetaData(router: Router): void {
        router.put(
            '/experiment/cluster-metadata', //TODO: Fix validation expressJoi(ValidationSchemas.SETCLUSTERMETADATA),
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
                    this.handleError(NNIError.FromError(err as any), res, true);
                }
        });
    }

    private listTrialJobs(router: Router): void {
        router.get('/trial-jobs', (req: Request, res: Response) => {
            this.nniManager.listTrialJobs(req.query['status'] as TrialJobStatus).then((jobInfos: TrialJobInfo[]) => {
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
            this.nniManager.getTrialJob(req.params['id']).then((jobDetail: TrialJobInfo) => {
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
            this.nniManager.cancelTrialJobByUser(req.params['id']).then(() => {
                res.send();
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getMetricData(router: Router): void {
        router.get('/metric-data/:job_id*?', async (req: Request, res: Response) => {
            this.nniManager.getMetricData(req.params['job_id'], req.query['type'] as MetricType).then((metricsData: MetricDataRecord[]) => {
                res.send(metricsData);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getMetricDataByRange(router: Router): void {
        router.get('/metric-data-range/:min_seq_id/:max_seq_id', async (req: Request, res: Response) => {
            const minSeqId = Number(req.params['min_seq_id']);
            const maxSeqId = Number(req.params['max_seq_id']);
            this.nniManager.getMetricDataByRange(minSeqId, maxSeqId).then((metricsData: MetricDataRecord[]) => {
                res.send(metricsData);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getLatestMetricData(router: Router): void {
        router.get('/metric-data-latest/', async (_req: Request, res: Response) => {
            this.nniManager.getLatestMetricData().then((metricsData: MetricDataRecord[]) => {
                res.send(metricsData);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getTrialFile(router: Router): void {
        router.get('/trial-file/:id/:filename', async(req: Request, res: Response) => {
            const filename = req.params['filename'];
            this.nniManager.getTrialFile(req.params['id'], filename).then((content: Buffer | string) => {
                const contentType = content instanceof Buffer ? 'application/octet-stream' : 'text/plain';
                res.header('Content-Type', contentType);
                if (content === '') {
                    content = `${filename} is empty.`;  // FIXME: this should be handled in front-end
                }
                res.send(content);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private exportData(router: Router): void {
        router.get('/export-data', (_req: Request, res: Response) => {
            this.nniManager.exportData().then((exportedData: string) => {
                res.send(exportedData);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getExperimentMetadata(router: Router): void {
        router.get('/experiment-metadata', (_req: Request, res: Response) => {
            Promise.all([
                this.nniManager.getExperimentProfile(),
                getExperimentsManager().getExperimentsInfo()
            ]).then(([profile, experimentInfo]) => {
                for (const info of experimentInfo as any) {
                    if (info.id === profile.id) {
                        res.send(info);
                        break;
                    }
                }
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private getExperimentsInfo(router: Router): void {
        router.get('/experiments-info', (_req: Request, res: Response) => {
            getExperimentsManager().getExperimentsInfo().then((experimentInfo: JSON) => {
                res.send(JSON.stringify(experimentInfo));
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private startTensorboardTask(router: Router): void {
        router.post('/tensorboard', (req: Request, res: Response) => {
            this.tensorboardManager.startTensorboardTask(req.body).then((taskDetail: TensorboardTaskInfo) => {
                this.log.info(taskDetail);
                res.send(Object.assign({}, taskDetail));
            }).catch((err: Error) => {
                this.handleError(err, res, false, 400);
            });
        });
    }

    private getTensorboardTask(router: Router): void {
        router.get('/tensorboard/:id', (req: Request, res: Response) => {
            this.tensorboardManager.getTensorboardTask(req.params['id']).then((taskDetail: TensorboardTaskInfo) => {
                res.send(Object.assign({}, taskDetail));
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private updateTensorboardTask(router: Router): void {
        router.put('/tensorboard/:id', (req: Request, res: Response) => {
            this.tensorboardManager.updateTensorboardTask(req.params['id']).then((taskDetail: TensorboardTaskInfo) => {
                res.send(Object.assign({}, taskDetail));
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private stopTensorboardTask(router: Router): void {
        router.delete('/tensorboard/:id', (req: Request, res: Response) => {
            this.tensorboardManager.stopTensorboardTask(req.params['id']).then((taskDetail: TensorboardTaskInfo) => {
                res.send(Object.assign({}, taskDetail));
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private stopAllTensorboardTask(router: Router): void {
        router.delete('/tensorboard-tasks', (_req: Request, res: Response) => {
            this.tensorboardManager.stopAllTensorboardTask().then(() => {
                res.send();
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private listTensorboardTask(router: Router): void {
        router.get('/tensorboard-tasks', (_req: Request, res: Response) => {
            this.tensorboardManager.listTensorboardTasks().then((taskDetails: TensorboardTaskInfo[]) => {
                res.send(taskDetails);
            }).catch((err: Error) => {
                this.handleError(err, res);
            });
        });
    }

    private stop(router: Router): void {
        router.delete('/experiment', (_req: Request, res: Response) => {
            res.send();
            globals.shutdown.initiate('REST request');
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

export function createRestHandler(): Router {
    return new NNIRestHandler().createRestHandler();
}
