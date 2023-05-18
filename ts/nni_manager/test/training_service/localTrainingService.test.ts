// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import chai from 'chai';
import chaiAsPromised from 'chai-as-promised';
import fs from 'fs';
import path from 'path';
import tmp from 'tmp';
import { TrialJobApplicationForm, TrialJobDetail} from '../../common/trainingService';
import { cleanupUnitTest, delay, prepareUnitTest, getExperimentRootDir } from '../../common/utils';
import { TrialConfigMetadataKey } from '../../training_service/common/trialConfigMetadataKey';
import { LocalTrainingService } from '../../training_service/local/localTrainingService';

// TODO: copy mockedTrail.py to local folder
const localCodeDir: string = tmp.dirSync().name.split('\\').join('\\\\');
const mockedTrialPath: string = './test/mock/mockedTrial.py'
fs.copyFileSync(mockedTrialPath, localCodeDir + '/mockedTrial.py')

describe('Unit Test for LocalTrainingService', () => {
    const config = <any>{
        platform: 'local',
        trialCommand: 'sleep 1h && echo hello',
        trialCodeDirectory: `${localCodeDir}`,
        trialGpuNumber: 0,  // TODO: add test case for gpu?
        maxTrialNumberPerGpu: 1,
        reuseMode: true,
    };

    const config2 = <any>{
        platform: 'local',
        trialCommand: 'python3 mockedTrial.py',
        trialCodeDirectory: `${localCodeDir}`,
        trialGpuNumber: 0,
        maxTrialNumberPerGpu: 1,
        reuseMode: true,
    };

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
    });

    it('List empty trial jobs', async () => {
        const localTrainingService = new LocalTrainingService(config);
        localTrainingService.run();

        //trial jobs should be empty, since there are no submitted jobs
        chai.expect(await localTrainingService.listTrialJobs()).to.be.empty;

        localTrainingService.cleanUp();
    });

    it('Submit job and Cancel job', async () => {
        const localTrainingService = new LocalTrainingService(config);
        localTrainingService.run();

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

        localTrainingService.cleanUp();
    }).timeout(20000);

    it('Get trial log', async () => {
        const localTrainingService = new LocalTrainingService(config);
        localTrainingService.run();

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
        chai.expect(await localTrainingService.getTrialFile(jobDetail.id, 'trial.log')).to.be.equals('trial log');
        chai.expect(await localTrainingService.getTrialFile(jobDetail.id, 'stderr')).to.be.equals('trial stderr');
        fs.unlinkSync(path.join(jobDetail.workingDirectory, 'trial.log'))
        fs.unlinkSync(path.join(jobDetail.workingDirectory, 'stderr'))
        fs.rmdirSync(jobDetail.workingDirectory)
        fs.rmdirSync(path.join(rootDir, 'trials'))

        await localTrainingService.cancelTrialJob(jobDetail.id);
        localTrainingService.cleanUp();
    }).timeout(20000);

    it('Read metrics, Add listener, and remove listener', async () => {
        const localTrainingService = new LocalTrainingService(config2);
        localTrainingService.run();

        // set meta data
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
        localTrainingService.cleanUp();
    }).timeout(20000);
});
