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
import { Deferred } from 'ts-deferred';

import * as component from '../common/component';
import { Database, DataStore, MetricData, MetricDataRecord, MetricType,
    TrialJobEvent, TrialJobEventRecord, TrialJobInfo, HyperParameterFormat,
    ExportedDataFormat } from '../common/datastore';
import { NNIError } from '../common/errors';
import { getExperimentId, isNewExperiment } from '../common/experimentStartupInfo';
import { getLogger, Logger } from '../common/log';
import { ExperimentProfile,  TrialJobStatistics } from '../common/manager';
import { TrialJobDetail, TrialJobStatus } from '../common/trainingService';
import { getDefaultDatabaseDir, mkDirP } from '../common/utils';

class NNIDataStore implements DataStore {
    private db: Database = component.get(Database);
    private log: Logger = getLogger();
    private initTask!: Deferred<void>;
    private multiPhase: boolean | undefined;

    public init(): Promise<void> {
        if (this.initTask !== undefined) {
            return this.initTask.promise;
        }
        this.initTask = new Deferred<void>();

        // TODO support specify database dir
        const databaseDir: string = getDefaultDatabaseDir();
        if(isNewExperiment()) {
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
            let n: number|undefined = map.get(value.status);
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

        return trialJobs[0];
    }

    public async storeMetricData(trialJobId: string, data: string): Promise<void> {
        const metrics: MetricData = JSON.parse(data);
        // REQUEST_PARAMETER is used to request new parameters for multiphase trial job,
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
        let exportedData: ExportedDataFormat[] = [];
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
                    let paraMap: Map<number, Object> = new Map();
                    let metricMap: Map<number, Object> = new Map();
                    for (const eachPara of job.hyperParameters) {
                        const parameters: HyperParameterFormat = <HyperParameterFormat>JSON.parse(eachPara);
                        paraMap.set(parameters.parameter_id, parameters.parameters);
                    }
                    for (const eachMetric of job.finalMetricData) {
                        const value: Object = JSON.parse(eachMetric.data);
                        metricMap.set(Number(eachMetric.parameterId), value);
                    }
                    paraMap.forEach((value: Object, key: number) => {
                        const metricValue: Object | undefined = metricMap.get(key);
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
        let importedData: string[] = [];
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
        const map: Map<string, TrialJobInfo> = this.getTrialJobsByReplayEvents(trialJobEvents);

        const finalMetricsMap: Map<string, MetricDataRecord[]> = await this.getFinalMetricData(trialJobId);

        for (const key of map.keys()) {
            const jobInfo: TrialJobInfo | undefined = map.get(key);
            if (jobInfo === undefined) {
                continue;
            }
            if (!(status !== undefined && jobInfo.status !== status)) {
                if (jobInfo.status === 'SUCCEEDED') {
                    jobInfo.finalMetricData = finalMetricsMap.get(jobInfo.id);
                }
                result.push(jobInfo);
            }
        }

        return result;
    }

    private async getFinalMetricData(trialJobId?: string): Promise<Map<string, MetricDataRecord[]>> {
        const map: Map<string, MetricDataRecord[]> = new Map();
        const metrics: MetricDataRecord[] = await this.getMetricData(trialJobId, 'FINAL');

        const multiPhase: boolean = await this.isMultiPhase();

        for (const metric of metrics) {
            const existMetrics: MetricDataRecord[] | undefined = map.get(metric.trialJobId);
            if (existMetrics !== undefined) {
                if (!multiPhase) {
                    this.log.error(`Found multiple FINAL results for trial job ${trialJobId}, metrics: ${JSON.stringify(metrics)}`);
                } else {
                    existMetrics.push(metric);
                }
            } else {
                map.set(metric.trialJobId, [metric]);
            }
        }

        return map;
    }

    private async isMultiPhase(): Promise<boolean> {
        if (this.multiPhase === undefined) {
            const expProfile: ExperimentProfile = await this.getExperimentProfile(getExperimentId());
            if (expProfile !== undefined) {
                this.multiPhase = expProfile.params.multiPhase;
            } else {
                return false;
            }
        }

        if (this.multiPhase !== undefined) {
            return this.multiPhase;
        } else {
            return false;
        }
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

    private parseHyperParameter(hParamStr: string): any {
        let hParam: any;
        try {
            hParam = JSON.parse(hParamStr);

            return hParam;
        } catch (err) {
            this.log.error(`Hyper parameter needs to be in json format: ${hParamStr}`);

            return undefined;
        }
    }

    // tslint:disable-next-line:cyclomatic-complexity
    private getTrialJobsByReplayEvents(trialJobEvents: TrialJobEventRecord[]):  Map<string, TrialJobInfo> {
        this.log.debug('getTrialJobsByReplayEvents begin');

        const map: Map<string, TrialJobInfo> = new Map();
        const hParamIdMap: Map<string, Set<number>> = new Map();

        // assume data is stored by time ASC order
        for (const record of trialJobEvents) {
            let jobInfo: TrialJobInfo | undefined;
            if (record.trialJobId === undefined || record.trialJobId.length < 1) {
                continue;
            }
            if (map.has(record.trialJobId)) {
                jobInfo = map.get(record.trialJobId);
            } else {
                jobInfo = {
                    id: record.trialJobId,
                    status: this.getJobStatusByLatestEvent('UNKNOWN', record.event),
                    hyperParameters: []
                };
            }
            if (!jobInfo) {
                throw new Error('Empty JobInfo');
            }
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
                const newHParam: any = this.parseHyperParameter(record.data);
                if (newHParam !== undefined) {
                    if (jobInfo.hyperParameters !== undefined) {
                        let hParamIds: Set<number> | undefined = hParamIdMap.get(jobInfo.id);
                        if (hParamIds === undefined) {
                            hParamIds = new Set();
                        }
                        if (!hParamIds.has(newHParam.parameter_index)) {
                            jobInfo.hyperParameters.push(JSON.stringify(newHParam));
                            hParamIds.add(newHParam.parameter_index);
                            hParamIdMap.set(jobInfo.id, hParamIds);
                        }
                    } else {
                        assert(false, 'jobInfo.hyperParameters is undefined');
                    }
                }
            }
            if (record.sequenceId !== undefined && jobInfo.sequenceId === undefined) {
                jobInfo.sequenceId = record.sequenceId;
            }
            map.set(record.trialJobId, jobInfo);
        }

        this.log.debug('getTrialJobsByReplayEvents done');

        return map;
    }
}

export { NNIDataStore };
