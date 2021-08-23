// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';
import * as fs from 'fs';
import * as path from 'path';
import * as tmp from 'tmp';
import * as component from '../../common/component';
import { TrialJobApplicationForm, TrialJobDetail} from '../../common/trainingService';
import { cleanupUnitTest, delay, prepareUnitTest, getExperimentRootDir } from '../../common/utils';
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
            sequenceId: 0,
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

    it('Get trial log', async () => {
        await localTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, trialConfig);

        // submit job
        const form: TrialJobApplicationForm = {
            sequenceId: 0,
            hyperParameters: {
                value: 'mock hyperparameters',
                index: 0
            }
        };

        const jobDetail: TrialJobDetail = await localTrainingService.submitTrialJob(form);

        // get trial log
        const rootDir: string = getExperimentRootDir()
        fs.mkdirSync(path.join(rootDir, 'trials'))
        fs.mkdirSync(jobDetail.workingDirectory)
        fs.writeFileSync(path.join(jobDetail.workingDirectory, 'trial.log'), 'trial log')
        fs.writeFileSync(path.join(jobDetail.workingDirectory, 'stderr'), 'trial stderr')
        chai.expect(await localTrainingService.getTrialLog(jobDetail.id, 'TRIAL_LOG')).to.be.equals('trial log');
        chai.expect(await localTrainingService.getTrialLog(jobDetail.id, 'TRIAL_ERROR')).to.be.equals('trial stderr');
        fs.unlinkSync(path.join(jobDetail.workingDirectory, 'trial.log'))
        fs.unlinkSync(path.join(jobDetail.workingDirectory, 'stderr'))
        fs.rmdirSync(jobDetail.workingDirectory)
        fs.rmdirSync(path.join(rootDir, 'trials'))

        await localTrainingService.cancelTrialJob(jobDetail.id);
    }).timeout(20000);

    it('Read metrics, Add listener, and remove listener', async () => {
        // set meta data
        const trialConfig: string = `{\"command\":\"python3 mockedTrial.py\", \"codeDir\":\"${localCodeDir}\",\"gpuNum\":0}`
        await localTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, trialConfig);

        // submit job
        const form: TrialJobApplicationForm = {
            sequenceId: 0,
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
