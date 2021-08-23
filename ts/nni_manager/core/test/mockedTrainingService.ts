// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Deferred } from 'ts-deferred';
import { Provider } from 'typescript-ioc';

import { MethodNotImplementedError } from '../../common/errors';
import { TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, LogType } from '../../common/trainingService';

const testTrainingServiceProvider: Provider = {
    get: () => { return new MockedTrainingService(); }
};

class MockedTrainingService extends TrainingService {
    public mockedMetaDataValue: string = "default";
    public jobDetail1: TrialJobDetail = {
        id: '1234',
        status: 'SUCCEEDED',
        submitTime: Date.now(),
        startTime: Date.now(),
        endTime: Date.now(),
        tags: ['test'],
        url: 'http://test',
        workingDirectory: '/tmp/mocked',
        form: {
            sequenceId: 0,
            hyperParameters: { value: '', index: 0 }
        },
    };
    public jobDetail2: TrialJobDetail = {
        id: '3456',
        status: 'SUCCEEDED',
        submitTime: Date.now(),
        startTime: Date.now(),
        endTime: Date.now(),
        tags: ['test'],
        url: 'http://test',
        workingDirectory: '/tmp/mocked',
        form: {
            sequenceId: 1,
            hyperParameters: { value: '', index: 1 }
        },
    };

    public listTrialJobs(): Promise<TrialJobDetail[]> {
        const deferred = new Deferred<TrialJobDetail[]>();

        deferred.resolve([this.jobDetail1, this.jobDetail2]);
        return deferred.promise;
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const deferred = new Deferred<TrialJobDetail>();
        if(trialJobId === '1234'){
            deferred.resolve(this.jobDetail1);
        }else if(trialJobId === '3456'){
            deferred.resolve(this.jobDetail2);
        }else{
            deferred.reject();
        }
        return deferred.promise;
    }

    public getTrialLog(trialJobId: string, logType: LogType): Promise<string> {
        throw new MethodNotImplementedError();
    }

    async run(): Promise<void> {

    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
    }

    public submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const deferred = new Deferred<TrialJobDetail>();
        return deferred.promise;
    }

    public updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const deferred = new Deferred<void>();
        if(trialJobId === '1234' || trialJobId === '3456'){
            deferred.resolve();
        }else{
            deferred.reject('job id error');
        }
        return deferred.promise;
    }

    public setClusterMetadata(key: string, value: string): Promise<void> {
        const deferred = new Deferred<void>();
        if(key == 'mockedMetadataKey'){
            this.mockedMetaDataValue = value;
            deferred.resolve();
        }else{
            deferred.reject('key error');
        }
        return deferred.promise;
    }

    public getClusterMetadata(key: string): Promise<string> {
        const deferred = new Deferred<string>();
        if(key == 'mockedMetadataKey'){
            deferred.resolve(this.mockedMetaDataValue);
        }else{
            deferred.reject('key error');
        }
        return deferred.promise;
    }

    public cleanUp(): Promise<void> {
        return Promise.resolve();
    }
}

export{MockedTrainingService, testTrainingServiceProvider}
