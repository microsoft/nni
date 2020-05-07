// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import { Deferred } from 'ts-deferred';

import * as component from '../common/component';
import {
    Database, DataStore, MetricData, MetricDataRecord, MetricType,
    TrialJobEvent, TrialJobEventRecord, TrialJobInfo, HyperParameterFormat,
    ExportedDataFormat
} from '../common/datastore';
import { NNIError } from '../common/errors';
import { isNewExperiment } from '../common/experimentStartupInfo';
import { getLogger, Logger } from '../common/log';
import { ExperimentProfile, TrialJobStatistics } from '../common/manager';
import { TrialJobDetail, TrialJobStatus } from '../common/trainingService';
import { getDefaultDatabaseDir, mkDirP } from '../common/utils';

class NNIDataStore implements DataStore {
    private db: Database = component.get(Database);
    private log: Logger = getLogger();
    private initTask!: Deferred<void>;

    public init(): Promise<void> {
        if (this.initTask !== undefined) {
            return this.initTask.promise;
        }
        this.initTask = new Deferred<void>();

        // TODO support specify database dir
        const databaseDir: string = getDefaultDatabaseDir();
        if (isNewExperiment()) {
            mkDirP(databaseDir).then(() => {
                this.db.init(true, databaseDir).then(() => {
                    this.log.info('Datastore initialization done');
                    this.initTask.resolve();
                }).catch((err: Error) => {
                    this.initTask.reject(err);
                });
            }).catch((err: Error) => {
                this.initTask.reject(err);
            });
        } else {
            this.db.init(false, databaseDir).then(() => {
                this.log.info('Datastore initialization done');
                this.initTask.resolve();
            }).catch((err: Error) => {
                this.initTask.reject(err);
            });
        }

        return this.initTask.promise;
    }

    public async close(): Promise<void> {
        await this.db.close();
    }

    public async storeExperimentProfile(experimentProfile: ExperimentProfile): Promise<void> {
        try {
            await this.db.storeExperimentProfile(experimentProfile);
        } catch (err) {
            throw NNIError.FromError(err, 'Datastore error: ');
        }
    }

    public getExperimentProfile(experimentId: string): Promise<ExperimentProfile> {
        return this.db.queryLatestExperimentProfile(experimentId);
    }

    public storeTrialJobEvent(
        event: TrialJobEvent, trialJobId: string, hyperParameter?: string, jobDetail?: TrialJobDetail): Promise<void> {
        this.log.debug(`storeTrialJobEvent: event: ${event}, data: ${hyperParameter}, jobDetail: ${JSON.stringify(jobDetail)}`);

        // Use the timestamp in jobDetail as TrialJobEvent timestamp for different events
        let timestamp: number | undefined;
        if (event === 'WAITING' && jobDetail) {
            timestamp = jobDetail.submitTime;
        } else if (event === 'RUNNING' && jobDetail) {
            timestamp = jobDetail.startTime;
        } else if (['EARLY_STOPPED', 'SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED'].includes(event) && jobDetail) {
            timestamp = jobDetail.endTime;
        }
        // Use current time as timestamp if timestamp is not assigned from jobDetail
        if (timestamp === undefined) {
            timestamp = Date.now();
        }

        return this.db.storeTrialJobEvent(event, trialJobId, timestamp, hyperParameter, jobDetail).catch(
            (err: Error) => {
                throw NNIError.FromError(err, 'Datastore error: ');
            }
        );
    }

    public async getTrialJobStatistics(): Promise<any[]> {
        const result: TrialJobStatistics[] = [];
        const jobs: TrialJobInfo[] = await this.listTrialJobs();
        const map: Map<TrialJobStatus, number> = new Map();

        jobs.forEach((value: TrialJobInfo) => {
            let n: number | undefined = map.get(value.status);
            if (!n) {
                n = 0;
            }
            map.set(value.status, n + 1);
        });

        map.forEach((value: number, key: TrialJobStatus) => {
            const statistics: TrialJobStatistics = {
                trialJobStatus: key,
                trialJobNumber: value
            };
            result.push(statistics);
        });

        return result;
    }

    public listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]> {
        return this.queryTrialJobs(status);
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobInfo> {
        const trialJobs: TrialJobInfo[] = await this.queryTrialJobs(undefined, trialJobId);
        assert(trialJobs.length <= 1);

        return trialJobs[0];
    }

    public async storeMetricData(trialJobId: string, data: string): Promise<void> {
        const metrics: MetricData = JSON.parse(data);
        // REQUEST_PARAMETER is used to request new parameters,
        // it is not metrics, so it is skipped here.
        if (metrics.type === 'REQUEST_PARAMETER') {
            return;
        }
        assert(trialJobId === metrics.trial_job_id);
        try {
            await this.db.storeMetricData(trialJobId, JSON.stringify({
                trialJobId: metrics.trial_job_id,
                parameterId: metrics.parameter_id,
                type: metrics.type,
                sequence: metrics.sequence,
                data: metrics.value,
                timestamp: Date.now()
            }));
        } catch (err) {
            throw NNIError.FromError(err, 'Datastore error');
        }
    }

    public getMetricData(trialJobId?: string, metricType?: MetricType): Promise<MetricDataRecord[]> {
        return this.db.queryMetricData(trialJobId, metricType);
    }

    public async exportTrialHpConfigs(): Promise<string> {
        const jobs: TrialJobInfo[] = await this.listTrialJobs();
        const exportedData: ExportedDataFormat[] = [];
        for (const job of jobs) {
            if (job.hyperParameters && job.finalMetricData) {
                if (job.hyperParameters.length === 1 && job.finalMetricData.length === 1) {
                    // optimization for non-multi-phase case
                    const parameters: HyperParameterFormat = <HyperParameterFormat>JSON.parse(job.hyperParameters[0]);
                    const oneEntry: ExportedDataFormat = {
                        parameter: parameters.parameters,
                        value: JSON.parse(job.finalMetricData[0].data),
                        id: job.id
                    };
                    exportedData.push(oneEntry);
                } else {
                    const paraMap: Map<number, Record<string, any>> = new Map();
                    const metricMap: Map<number, Record<string, any>> = new Map();
                    for (const eachPara of job.hyperParameters) {
                        const parameters: HyperParameterFormat = <HyperParameterFormat>JSON.parse(eachPara);
                        paraMap.set(parameters.parameter_id, parameters.parameters);
                    }
                    for (const eachMetric of job.finalMetricData) {
                        const value: Record<string, any> = JSON.parse(eachMetric.data);
                        metricMap.set(Number(eachMetric.parameterId), value);
                    }
                    paraMap.forEach((value: Record<string, any>, key: number) => {
                        const metricValue: Record<string, any> | undefined = metricMap.get(key);
                        if (metricValue) {
                            const oneEntry: ExportedDataFormat = {
                                parameter: value,
                                value: metricValue,
                                id: job.id
                            };
                            exportedData.push(oneEntry);
                        }
                    });
                }
            }
        }

        return JSON.stringify(exportedData);
    }

    public async getImportedData(): Promise<string[]> {
        const importedData: string[] = [];
        const importDataEvents: TrialJobEventRecord[] = await this.db.queryTrialJobEvent(undefined, 'IMPORT_DATA');
        for (const event of importDataEvents) {
            if (event.data) {
                importedData.push(event.data);
            }
        }

        return importedData;
    }

    private async queryTrialJobs(status?: TrialJobStatus, trialJobId?: string): Promise<TrialJobInfo[]> {
        const result: TrialJobInfo[] = [];
        const trialJobEvents: TrialJobEventRecord[] = await this.db.queryTrialJobEvent(trialJobId);
        if (trialJobEvents === undefined) {
            return result;
        }
        const trialMap: Map<string, TrialJobInfo> = this.getTrialsByReplayEvents(trialJobEvents);

        const finalMetricsMap: Map<string, MetricDataRecord[]> = await this.getFinalMetricData(trialJobId);

        for (const key of trialMap.keys()) {
            const trialInfo: TrialJobInfo | undefined = trialMap.get(key);
            if (trialInfo === undefined) {
                continue;
            }
            if (!(status !== undefined && trialInfo.status !== status)) {
                if (trialInfo.status === 'SUCCEEDED') {
                    trialInfo.finalMetricData = finalMetricsMap.get(trialInfo.id);
                }
                result.push(trialInfo);
            }
        }

        return result;
    }

    private async getFinalMetricData(trialJobId?: string): Promise<Map<string, MetricDataRecord[]>> {
        const map: Map<string, MetricDataRecord[]> = new Map();
        const metrics: MetricDataRecord[] = await this.getMetricData(trialJobId, 'FINAL');

        for (const metric of metrics) {
            const existMetrics: MetricDataRecord[] | undefined = map.get(metric.trialJobId);
            if (existMetrics !== undefined) {
                existMetrics.push(metric);
            } else {
                map.set(metric.trialJobId, [metric]);
            }
        }

        return map;
    }

    private getJobStatusByLatestEvent(oldStatus: TrialJobStatus, event: TrialJobEvent): TrialJobStatus {
        switch (event) {
            case 'USER_TO_CANCEL':
                return 'USER_CANCELED';
            case 'ADD_CUSTOMIZED':
                return 'WAITING';
            case 'ADD_HYPERPARAMETER':
                return oldStatus;
            default:
        }

        return <TrialJobStatus>event;
    }

    private parseMetricData(metricDataString: string): any {
        let metricData: any;
        try {
            metricData = JSON.parse(metricDataString);

            return metricData;
        } catch (err) {
            this.log.error(`Metric data needs to be in json format: ${metricDataString}`);

            return undefined;
        }
    }

    private getTrialsByReplayEvents(trialJobEvents: TrialJobEventRecord[]): Map<string, TrialJobInfo> {
        this.log.debug('getTrialsByReplayEvents begin');

        // For compatiable, use same structure for job and trial. And will return trials.
        const jobMap: Map<string, TrialJobInfo> = new Map();
        const trialMap: Map<string, TrialJobInfo> = new Map();
        const jobTrialMap: Map<string, string[]> = new Map();

        // assume data is stored by time ASC order
        for (const record of trialJobEvents) {
            let jobInfo: TrialJobInfo | undefined;
            if (record.trialJobId === undefined || record.trialJobId.length < 1) {
                continue;
            }
            if (jobMap.has(record.trialJobId)) {
                jobInfo = jobMap.get(record.trialJobId);
            } else {
                jobInfo = {
                    id: record.trialJobId,
                    status: this.getJobStatusByLatestEvent('UNKNOWN', record.event),
                    hyperParameters: []
                };
                jobMap.set(record.trialJobId, jobInfo);
            }
            if (!jobInfo) {
                throw new Error('Empty JobInfo');
            }
            /* eslint-disable no-fallthrough */
            switch (record.event) {
                case 'RUNNING':
                    if (record.timestamp !== undefined) {
                        jobInfo.startTime = record.timestamp;
                    }
                case 'WAITING':
                    if (record.logPath !== undefined) {
                        jobInfo.logPath = record.logPath;
                    }
                    // Initially assign WAITING timestamp as job's start time,
                    // If there is RUNNING state event, it will be updated as RUNNING state timestamp
                    if (jobInfo.startTime === undefined && record.timestamp !== undefined) {
                        jobInfo.startTime = record.timestamp;
                    }
                    break;
                case 'SUCCEEDED':
                case 'FAILED':
                case 'USER_CANCELED':
                case 'SYS_CANCELED':
                case 'EARLY_STOPPED':
                    if (record.logPath !== undefined) {
                        jobInfo.logPath = record.logPath;
                    }
                    jobInfo.endTime = record.timestamp;
                    if (jobInfo.startTime === undefined && record.timestamp !== undefined) {
                        jobInfo.startTime = record.timestamp;
                    }
                default:
            }
            jobInfo.status = this.getJobStatusByLatestEvent(jobInfo.status, record.event);

            if (record.data !== undefined && record.data.trim().length > 0) {
                const metricData: any = this.parseMetricData(record.data);
                if (metricData !== undefined) {
                    // update trials here.
                    const trialId = this.getTrialId(jobInfo.id, metricData);
                    let trialInfo: TrialJobInfo | undefined;
                    if (trialMap.has(trialId)) {
                        trialInfo = trialMap.get(trialId);
                        if (trialInfo && trialInfo.hyperParameters) {
                            trialInfo.hyperParameters.push(record.data);
                        }
                    } else {
                        trialInfo = {
                            id: trialId,
                            status: this.getJobStatusByLatestEvent('RUNNING', record.event),
                            hyperParameters: [record.data],
                            startTime: record.timestamp,
                            sequenceId: record.sequenceId
                        };
                        trialMap.set(trialId, trialInfo);
                    }
                    if (!trialInfo) {
                        throw new Error('Empty trialInfo');
                    }

                    let trialIds = jobTrialMap.get(jobInfo.id);
                    if (trialIds === undefined) {
                        trialIds = [];
                        jobTrialMap.set(jobInfo.id, trialIds);
                    }
                    trialIds.push(trialId);
                }
            }
        }

        for (const jobTrial of jobTrialMap) {
            const jobInfo = jobMap.get(jobTrial[0]);
            const trialIds = jobTrial[1];
            let trialInfo: TrialJobInfo | undefined;
            let lastTrialInfo: TrialJobInfo | undefined;

            if (!jobInfo) {
                throw new Error('Empty jobInfo');
            }

            for (const trialId of trialIds) {
                trialInfo = trialMap.get(trialId);
                if (!trialInfo) {
                    throw new Error('Empty trialInfo');
                }
                if (lastTrialInfo) {
                    lastTrialInfo.endTime = trialInfo.startTime;
                }
                trialInfo.logPath = jobInfo.logPath;
                // There is no status reported for each trial. So assume it's success, if it's not last one.
                // The last trial status is the same as job.
                trialInfo.status = 'SUCCEEDED';

                lastTrialInfo = trialInfo;
            }
            // the last one should follow job information
            if (trialInfo) {
                trialInfo.endTime = jobInfo.endTime;
                trialInfo.status = jobInfo.status;
            }
        }
        this.log.debug('getTrialsByReplayEvents done');

        return trialMap;
    }

    private getTrialId(jobId: string, metricData: MetricData): string {
        const trialId = jobId;
        const parameterId = metricData.parameter_id;
        return `${trialId}-${parameterId}`;
    }
}

export { NNIDataStore };
