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
    TrialJobEvent, TrialJobEventRecord, TrialJobInfo } from '../common/datastore';
import { isNewExperiment } from '../common/experimentStartupInfo';
import { getExperimentId } from '../common/experimentStartupInfo';
import { getLogger, Logger } from '../common/log';
import { ExperimentProfile,  TrialJobStatistics } from '../common/manager';
import { TrialJobStatus } from '../common/trainingService';
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
                    this.initTask.resolve();
                }).catch((err: Error) => {
                    this.initTask.reject(err);
                });
            }).catch((err: Error) => {
                this.initTask.reject(err);
            });
        } else {
            this.db.init(false, databaseDir).then(() => {
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
        await this.db.storeExperimentProfile(experimentProfile);
    }

    public getExperimentProfile(experimentId: string): Promise<ExperimentProfile> {
        return this.db.queryLatestExperimentProfile(experimentId);
    }

    public storeTrialJobEvent(event: TrialJobEvent, trialJobId: string, data?: string, logPath?: string): Promise<void> {
        this.log.debug(`storeTrialJobEvent: event: ${event}, data: ${data}, logpath: ${logPath}`);

        return this.db.storeTrialJobEvent(event, trialJobId, data, logPath);
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
        await this.db.storeMetricData(trialJobId, JSON.stringify({
            trialJobId: metrics.trial_job_id,
            parameterId: metrics.parameter_id,
            type: metrics.type,
            sequence: metrics.sequence,
            data: metrics.value,
            timestamp: Date.now()
        }));
    }

    public getMetricData(trialJobId: string, metricType: MetricType): Promise<MetricDataRecord[]> {
        return this.db.queryMetricData(trialJobId, metricType);
    }

    private async queryTrialJobs(status?: TrialJobStatus, trialJobId?: string): Promise<TrialJobInfo[]> {
        const result: TrialJobInfo[]= [];
        const trialJobEvents: TrialJobEventRecord[] = await this.db.queryTrialJobEvent(trialJobId);
        if (trialJobEvents === undefined) {
            return result;
        }
        const map: Map<string, TrialJobInfo> = this.getTrialJobsByReplayEvents(trialJobEvents);

        for (let key of map.keys()) {
            const jobInfo = map.get(key);
            if (jobInfo === undefined) {
                continue;
            }
            if (!(status !== undefined && jobInfo.status !== status)) {
                if (jobInfo.status === 'SUCCEEDED') {
                    jobInfo.finalMetricData = await this.getFinalMetricData(jobInfo.id);
                }
                result.push(jobInfo);
            }
        }

        return result;
    }

    private async getFinalMetricData(trialJobId: string): Promise<any> {
        const metrics: MetricDataRecord[] = await this.getMetricData(trialJobId, 'FINAL');

        const multiPhase: boolean = await this.isMultiPhase();

        if (metrics.length > 1 && !multiPhase) {
            this.log.error(`Found multiple FINAL results for trial job ${trialJobId}`);
        }

        return metrics[metrics.length - 1];
    }

    private async isMultiPhase(): Promise<boolean> {
        if (this.multiPhase === undefined) {
            this.multiPhase = (await this.getExperimentProfile(getExperimentId())).params.multiPhase;
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

    private mergeHyperParameters(hyperParamList: string[], newParamStr: string): string[] {
        const mergedHyperParams: any[] = [];
        const newParam: any = JSON.parse(newParamStr);
        for (const hyperParamStr of hyperParamList) {
            const hyperParam: any = JSON.parse(hyperParamStr);
            mergedHyperParams.push(hyperParam);
        }
        if (mergedHyperParams.filter((value: any) => value.parameter_index === newParam.parameter_index).length <= 0) {
            mergedHyperParams.push(newParam);
        }

        return mergedHyperParams.map<string>((value: any) => { return JSON.stringify(value); });
    }

    private getTrialJobsByReplayEvents(trialJobEvents: TrialJobEventRecord[]):  Map<string, TrialJobInfo> {
        const map: Map<string, TrialJobInfo> = new Map();
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
                    break;
                case 'SUCCEEDED':
                case 'FAILED':
                case 'USER_CANCELED':
                case 'SYS_CANCELED':
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
                if (jobInfo.hyperParameters !== undefined) {
                    jobInfo.hyperParameters = this.mergeHyperParameters(jobInfo.hyperParameters, record.data);
                } else {
                    assert(false, 'jobInfo.hyperParameters is undefined');
                }
            }
            map.set(record.trialJobId, jobInfo);
        }

        return map;
    }
}

export { NNIDataStore };
