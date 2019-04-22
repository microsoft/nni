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

import * as assert from 'assert';
import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';
import * as fs from 'fs';
import * as tmp from 'tmp';
import * as component from '../../common/component';
import { TrialJobApplicationForm, TrialJobDetail, TrainingService } from '../../common/trainingService';
import { cleanupUnitTest, delay, prepareUnitTest } from '../../common/utils';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { LocalTrainingService } from '../local/localTrainingService';

// TODO: copy mockedTrail.py to local folder
const localCodeDir: string = tmp.dirSync().name.split('\\').join('\\\\');
const mockedTrialPath: string = './training_service/test/mockedTrial.py'
fs.copyFileSync(mockedTrialPath, localCodeDir + '/mockedTrial.py')

describe('Unit Test for LocalTrainingService', () => {
    let trialConfig: any = `{"command":"sleep 1h && echo hello","codeDir":"${localCodeDir}","gpuNum":1}`

    let localTrainingService: LocalTrainingService;

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
    });

    beforeEach(() => {
        localTrainingService = component.get(LocalTrainingService);
        localTrainingService.run();
    });

    afterEach(() => {
        localTrainingService.cleanUp();
    });

    it('List empty trial jobs', async () => {
        //trial jobs should be empty, since there are no submitted jobs
        chai.expect(await localTrainingService.listTrialJobs()).to.be.empty;
    });
    
    it('setClusterMetadata and getClusterMetadata', async () => {
        await localTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, trialConfig);
        localTrainingService.getClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG).then((data)=>{
            chai.expect(data).to.be.equals(trialConfig);
        });
    });

    it('Submit job and Cancel job', async () => {
        await localTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, trialConfig);

        // submit job
        const form: TrialJobApplicationForm = {
            jobType: 'TRIAL',
            hyperParameters: {
                value: 'mock hyperparameters',
                index: 0
            }
        };
        const jobDetail: TrialJobDetail = await localTrainingService.submitTrialJob(form);
        chai.expect(jobDetail.status).to.be.equals('WAITING');
        await localTrainingService.cancelTrialJob(jobDetail.id);
        chai.expect(jobDetail.status).to.be.equals('USER_CANCELED');
    }).timeout(20000);
    
    it('Read metrics, Add listener, and remove listener', async () => {
        // set meta data
        const trialConfig: string = `{\"command\":\"python3 mockedTrial.py\", \"codeDir\":\"${localCodeDir}\",\"gpuNum\":0}`
        await localTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, trialConfig);

        // submit job
        const form: TrialJobApplicationForm = {
            jobType: 'TRIAL',
            hyperParameters: {
                value: 'mock hyperparameters',
                index: 0
            }
        };
        const jobDetail: TrialJobDetail = await localTrainingService.submitTrialJob(form);
        chai.expect(jobDetail.status).to.be.equals('WAITING');
        localTrainingService.listTrialJobs().then((jobList)=>{
            chai.expect(jobList.length).to.be.equals(1);
        });
        // Add metrics listeners
        const listener1 = function f1(metric: any) {
            chai.expect(metric.id).to.be.equals(jobDetail.id);
        }
        localTrainingService.addTrialJobMetricListener(listener1);
        // Wait to collect metric
        await delay(1000);

        await localTrainingService.cancelTrialJob(jobDetail.id);
        localTrainingService.removeTrialJobMetricListener(listener1);
    }).timeout(20000);

    it('Test multiphaseSupported', () => {
        chai.expect(localTrainingService.isMultiPhaseJobSupported).to.be.equals(true)
    })
});