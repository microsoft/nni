/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.

 * MIT License

 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import { Deferred } from 'ts-deferred';
import { Provider } from 'typescript-ioc';

import { MethodNotImplementedError } from '../../common/errors';
import { TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric } from '../../common/trainingService';

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
        form: { // FIXME
            sequenceId: 0,
            hyperParameters: {
                value: '',
                index: 0,
            },
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
            sequenceId: 0,
            hyperParameters: {
                value: '',
                index: 0,
            },
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
