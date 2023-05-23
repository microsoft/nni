// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { assert } from 'chai';
import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';

import { MethodNotImplementedError } from '../../common/errors';
import { TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric } from '../../common/trainingService';

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
    {id: '7890', status: 'WAITING'}];

// the first two are the resumed trials
// the last three are newly submitted, among which there is one customized trial
const idStatusListResume = [
    {id: '5678', status: 'RUNNING'},
    {id: '7890', status: 'RUNNING'},
    {id: '9012', status: 'RUNNING'},
    {id: '1011', status: 'RUNNING'},
    {id: '1112', status: 'RUNNING'}];

export class MockedTrainingService implements TrainingService {
    private readonly eventEmitter: EventEmitter;
    private mockedMetaDataValue: string = "default";
    private jobDetailList: Map<string, TrialJobDetail>;
    private mode: string;
    private submittedCnt: number = 0;

    constructor(mode: string = '') {
        this.eventEmitter = new EventEmitter();
        this.mode = mode;
        this.jobDetailList = new Map<string, TrialJobDetail>();
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
        if (this.mode === 'create_stage') {
            assert(this.submittedCnt < idStatusList.length);
            const submittedOne: TrialJobDetail = Object.assign({},
                jobDetailTemplate, idStatusList[this.submittedCnt],
                {submitTime: Date.now(), startTime: Date.now(), form: _form});
            this.jobDetailList.set(submittedOne.id, submittedOne);
            this.submittedCnt++;
            // only update the first two trials to SUCCEEDED
            if (['1234', '3456'].includes(submittedOne.id)) {
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
                }, 100);
                setTimeout(() => {
                    this.jobDetailList.set(submittedOne.id, Object.assign({}, submittedOne, {endTime: Date.now(), status: 'SUCCEEDED'}));
                }, 150);
            }
            return Promise.resolve(submittedOne);
        }
        else if (this.mode === 'resume_stage') {
            assert(this.submittedCnt < idStatusListResume.length);
            const submittedOne: TrialJobDetail = Object.assign({},
                jobDetailTemplate, idStatusListResume[this.submittedCnt],
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
            }, 100);
            setTimeout(() => {
                this.jobDetailList.set(submittedOne.id, Object.assign({}, submittedOne, {endTime: Date.now(), status: 'SUCCEEDED'}));
            }, 150);
            return Promise.resolve(submittedOne);
        }
        else {
            throw new Error('Unknown mode for the mocked training service!');
        }
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
