// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { assert } from 'console';
import * as fs from 'fs';
import { Deferred } from 'ts-deferred';

import { DataStore, MetricData, MetricDataRecord, MetricType,
    TrialJobEvent, TrialJobEventRecord, TrialJobInfo } from '../../common/datastore';
import { ExperimentProfile,  TrialJobStatistics } from '../../common/manager';
import { TrialJobStatus } from '../../common/trainingService';

class SimpleDb {
    private name: string = '';
    private fileName: string = '';

    private db: Array<any> = new Array();
    private map: Map<string, number> = new Map<string, number>();  // map key to data index

    constructor (name: string, filename: string) {
        this.name = name;
        this.fileName = filename;
    }

    async saveData(data: any, key?: string): Promise<void> {
        let index;
        if (key && this.map.has(key)) {
            index = this.map.get(key);
        }

        if (index === undefined) {
            index = this.db.push(data) - 1;
        } else {
            this.db[index] = data;
        }

        if (key) {
            this.map.set(key, index);
        }
        await this.persist();
    }

    listAllData(): Promise<Array<any>> {
        const deferred = new Deferred<Array<any>>();
        deferred.resolve(this.db);

        return deferred.promise;
    }

    getData(key: string): Promise<any> {
        const deferred = new Deferred<any>();
        if (this.map.has(key)) {
            const index = this.map.get(key);
            if(index !== undefined && index >= 0) {
                deferred.resolve(this.db[index]);
            } else {
                deferred.reject(new Error(`Key or index not found: ${this.name}, ${key}`));
            }
        } else {
            console.log(`Key not found: ${this.name}, ${key}`);
            deferred.resolve(undefined);
        }
        return deferred.promise;
    }

    persist(): Promise<void> {
        const deferred = new Deferred<void>();
        fs.writeFileSync(this.fileName, JSON.stringify({
            name: this.name,
            data: this.db,
            index: JSON.stringify([...this.map])
        }, null, 4));
        deferred.resolve();
        return deferred.promise;
    }
}

class MockedDataStore implements DataStore {

    private dbExpProfile: SimpleDb = new SimpleDb('exp_profile', './exp_profile.json');
    private dbTrialJobs: SimpleDb = new SimpleDb('trial_jobs', './trial_jobs.json');
    private dbMetrics: SimpleDb = new SimpleDb('metrics', './metrics.json');

    trailJob1 = {
        event: 'ADD_CUSTOMIZED',
        timestamp: Date.now(),
        trialJobId: "4321",
        data: ''
    }

    metrics1 = {
        timestamp: Date.now(),
        trialJobId: '4321',
        parameterId: 'param1',
        type: 'CUSTOM',
        sequence: 21,
        data: ''
    }

    init(): Promise<void> {
        this.dbTrialJobs.saveData(this.trailJob1);
        this.dbMetrics.saveData(this.metrics1);
        return Promise.resolve();
    }

    close(): Promise<void> {
        return Promise.resolve();
    }

    async storeExperimentProfile(experimentProfile: ExperimentProfile): Promise<void> {
        await this.dbExpProfile.saveData(experimentProfile, experimentProfile.id);
    }

    async getExperimentProfile(experimentId: string): Promise<ExperimentProfile> {
        return await this.dbExpProfile.getData(experimentId);
    }

    async storeTrialJobEvent(event: TrialJobEvent, trialJobId: string, data?: string | undefined): Promise<void> {
        const dataRecord: TrialJobEventRecord = {
            event: event,
            timestamp: Date.now(),
            trialJobId: trialJobId,
            data: data
        }
        await this.dbTrialJobs.saveData(dataRecord);
    }

    async getTrialJobStatistics(): Promise<any[]> {
        const result: TrialJobStatistics[] = [];
        const jobs = await this.listTrialJobs();
        const map: Map<TrialJobStatus, number> = new Map();

        jobs.forEach((value) => {
            let n: number|undefined = map.get(value.status);
            if (!n) {
                n = 0;
            }
            map.set(value.status, n + 1);
        })

        map.forEach((value, key) => {
            const statistics: TrialJobStatistics = {
                trialJobStatus: key,
                trialJobNumber: value
            }
            result.push(statistics);
        })
        return result;
    }

    async listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]> {
        const trialJobEvents: TrialJobEventRecord[] = await this.dbTrialJobs.listAllData();
        const map: Map<string, TrialJobInfo> = this.getTrialJobsByReplayEvents(trialJobEvents);
        const result: TrialJobInfo[]= [];
        for (let key of map.keys()) {
            const jobInfo = map.get(key);
            if (jobInfo === undefined) {
                continue;
            }
            if (!(status && jobInfo.status !== status)) {
                if (jobInfo.status === 'SUCCEEDED') {
                    jobInfo.finalMetricData = await this.getFinalMetricData(jobInfo.trialJobId);
                }
                result.push(jobInfo);
            }
        }
        return result;
    }

    async storeMetricData(trialJobId: string, data: string): Promise<void> {
        const metrics = JSON.parse(data) as MetricData;
        assert(trialJobId === metrics.trial_job_id);
        await this.dbMetrics.saveData({
            trialJobId: metrics.trial_job_id,
            parameterId: metrics.parameter_id,
            type: metrics.type,
            data: metrics.value,
            timestamp: Date.now()
        });
    }

    async getMetricData(trialJobId: string, metricType: MetricType): Promise<MetricDataRecord[]> {
        const result: MetricDataRecord[] = []
        const allMetrics = await this.dbMetrics.listAllData();
        allMetrics.forEach((value) => {
            const metrics = <MetricDataRecord>value;
            if (metrics.type === metricType && metrics.trialJobId === trialJobId) {
                result.push(metrics);
            }
        });

        return result;
    }

    async exportTrialHpConfigs(): Promise<string> {
        const ret: string = '';
        return Promise.resolve(ret);
    }

    async getImportedData(): Promise<string[]> {
        const ret: string[] = [];
        return Promise.resolve(ret);
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobInfo> {
        return Promise.resolve({
            trialJobId: '1234',
            status: 'SUCCEEDED',
            startTime: Date.now(),
            endTime: Date.now()
        });
    }

    private async getFinalMetricData(trialJobId: string): Promise<any> {
        const metrics: MetricDataRecord[] = await this.getMetricData(trialJobId, "FINAL");
        assert(metrics.length <= 1);
        if (metrics.length == 1) {
            return metrics[0];
        } else {
            return undefined;
        }
    }

    private getJobStatusByLatestEvent(event: TrialJobEvent): TrialJobStatus {
        switch(event) {
            case 'USER_TO_CANCEL':
                return 'USER_CANCELED';
            case 'ADD_CUSTOMIZED':
                return 'WAITING';
        }
        return <TrialJobStatus>event;
    }

    private getTrialJobsByReplayEvents(trialJobEvents: TrialJobEventRecord[]):  Map<string, TrialJobInfo> {
        const map: Map<string, TrialJobInfo> = new Map();
        // assume data is stored by time ASC order
        for (let record of trialJobEvents) {
            let jobInfo: TrialJobInfo | undefined;
            if (map.has(record.trialJobId)) {
                jobInfo = map.get(record.trialJobId);
            } else {
                jobInfo = {
                    trialJobId: record.trialJobId,
                    status: this.getJobStatusByLatestEvent(record.event),
                };
            }
            if (!jobInfo) {
                throw new Error('Empty JobInfo');
            }
            switch (record.event) {
                case 'RUNNING':
                    jobInfo.startTime = Date.now();
                    break;
                case 'SUCCEEDED':
                case 'FAILED':
                case 'USER_CANCELED':
                case 'SYS_CANCELED':
                case 'EARLY_STOPPED':
                    jobInfo.endTime = Date.now();
            }
            jobInfo.status = this.getJobStatusByLatestEvent(record.event);
            map.set(record.trialJobId, jobInfo);
        }
        return map;
    }
}

export { MockedDataStore };
