// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import internal from 'stream';
import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { Provider } from 'typescript-ioc';

import { MethodNotImplementedError } from '../../common/errors';
import { TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric } from '../../common/trainingService';

const testTrainingServiceProvider: Provider = {
    get: () => { return new MockedTrainingService(''); }
};

const jobDetailTemplate: TrialJobDetail = {
    id: 'xxxx',
    status: 'WAITING',
    submitTime: Date.now(),
    startTime: undefined,
    endTime: undefined,
    tags: ['test'],
    url: 'http://test',
    workingDirectory: '/tmp/mocked',
    form: {
        sequenceId: 0,
        hyperParameters: { value: '', index: 0 }
    },
};

const idStatusList = [
    {id: '1234', status: 'RUNNING'},
    {id: '3456', status: 'RUNNING'},
    {id: '5678', status: 'RUNNING'},
    {id: '7890', status: 'WAITING'},
    {id: '5678', status: 'SUCCEEDED'},
    {id: '7890', status: 'SUCCEEDED'}];

// let jobDetail1: TrialJobDetail = Object.assign({},
//     jobDetailTemplate, {id: '1234', status: 'SUCCEEDED'});

// let jobDetail2: TrialJobDetail = Object.assign({},
//     jobDetailTemplate, {id: '3456', status: 'SUCCEEDED'});

// let jobDetail3: TrialJobDetail = Object.assign({},
//     jobDetailTemplate, {id: '5678', status: 'RUNNING'});

// let jobDetail4: TrialJobDetail = Object.assign({},
//     jobDetailTemplate, {id: '7890', status: 'WAITING'});

// let jobDetail3updated: TrialJobDetail = Object.assign({},
//     jobDetailTemplate, {id: '5678', status: 'SUCCEEDED'});

// let jobDetail4updated: TrialJobDetail = Object.assign({},
//     jobDetailTemplate, {id: '7890', status: 'SUCCEEDED'});

class MockedTrainingService implements TrainingService {
    private readonly eventEmitter: EventEmitter;
    private mockedMetaDataValue: string = "default";
    private jobDetailList: Map<string, TrialJobDetail>;
    // private mode: string;
    private submittedCnt: number = 0;

    constructor(mode: string) {
        this.eventEmitter = new EventEmitter();
        // this.mode = mode;
        this.jobDetailList = new Map<string, TrialJobDetail>();
        if (mode === 'create_stage') {
            // this.jobDetailList.push(jobDetail1);
            // this.jobDetailList.push(jobDetail2);
            // this.jobDetailList.push(jobDetail3);
            // this.jobDetailList.push(jobDetail4);
        }
        else if (mode === 'resume_stage') {
            // this.jobDetailList.push(jobDetail3updated);
            // this.jobDetailList.push(jobDetail4updated);
        }
    }

    public listTrialJobs(): Promise<TrialJobDetail[]> {
        const trialJobs: TrialJobDetail[] = Array.from(this.jobDetailList.values());
        return Promise.resolve(trialJobs);
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const jobDetail: TrialJobDetail | undefined = this.jobDetailList.get(trialJobId);
        if (jobDetail !== undefined) {
            return Promise.resolve(jobDetail);
        }
        else {
            return Promise.reject('job id error');
        }
    }

    public getTrialFile(_trialJobId: string, _fileName: string): Promise<string> {
        throw new MethodNotImplementedError();
    }

    async run(): Promise<void> {

    }

    public addTrialJobMetricListener(_listener: (_metric: TrialJobMetric) => void): void {
        this.eventEmitter.on('metric', _listener);
    }

    public removeTrialJobMetricListener(_listener: (_metric: TrialJobMetric) => void): void {
        this.eventEmitter.off('metric', _listener);
    }

    public submitTrialJob(_form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const submittedOne: TrialJobDetail = Object.assign({},
            jobDetailTemplate, idStatusList[this.submittedCnt],
            {submitTime: Date.now(), startTime: Date.now(), form: _form});
        this.jobDetailList.set(submittedOne.id, submittedOne);
        this.submittedCnt++;
        // Emit metric data here for simplicity
        // Set timeout to make sure when the metric is received by nnimanager,
        // the corresponding trial job exists.
        setTimeout(() => {
            this.eventEmitter.emit('metric', {
                id: submittedOne.id,
                data: JSON.stringify({
                    'parameter_id': JSON.parse(submittedOne.form.hyperParameters.value)['parameter_id'],
                    'trial_job_id': submittedOne.id,
                    'type': 'FINAL',
                    'sequence': 0,
                    'value': '0.9'})
            });
        }, 200);
        // only update the first two trials to SUCCEEDED
        if (['1234', '3456'].includes(submittedOne.id)) {
            setTimeout(() => {
                this.jobDetailList.set(submittedOne.id, Object.assign({}, submittedOne, {endTime: Date.now(), status: 'SUCCEEDED'}));
            }, 1000);
        }
        return Promise.resolve(submittedOne);
    }

    public updateTrialJob(_trialJobId: string, _form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public cancelTrialJob(trialJobId: string, _isEarlyStopped: boolean = false): Promise<void> {
        if (this.jobDetailList.has(trialJobId))
            return Promise.resolve();
        else
            return Promise.reject('job id error');
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

    public getTrialOutputLocalPath(_trialJobId: string): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public fetchTrialOutput(_trialJobId: string, _subpath: string): Promise<void> {
        throw new MethodNotImplementedError();
    }
}

export{MockedTrainingService, testTrainingServiceProvider}
