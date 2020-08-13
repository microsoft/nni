// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';
import * as fs from 'fs';
import * as tmp from 'tmp';
import * as component from '../../common/component';
import { TrialJobApplicationForm, TrialJobDetail, TrainingService } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { AdlTrainingService } from '../kubernetes/adl/adlTrainingService';

// TODO: copy mockedTrail.py to local folder
const localCodeDir: string = tmp.dirSync().name
const mockedTrialPath: string = './training_service/test/mockedTrial.py'
fs.copyFileSync(mockedTrialPath, localCodeDir + '/mockedTrial.py')

describe('Unit Test for AdlTrainingService', () => {
    let skip: boolean = false;
    let testAdlTrialConfig: any = JSON.stringify({
        "command": "python3 /root/apps/nni_linear_regression/main.py",
        "codeDir": ".",
        "gpuNum": 0,
        "image": "registry.petuum.com/dev/adaptdl-submit:latest",
        "checkpoint": {
            "storageClass": "dfs",
            "storageSize": "1Gi"
        }
    });
    let testAdlTrialConfig2: any = JSON.stringify({
        "command": "python3 /root/apps/nni_linear_regression/main.py",
        "codeDir": ".",
        "gpuNum": 0,
        "image": "registry.petuum.com/dev/adaptdl-submit:latest",
        "imagePullSecrets": [
            {
                "name": "stagingsecrets"
            }
        ],
        "checkpoint": {
            "storageClass": "dfs",
            "storageSize": "1Gi"
        },
        "nfs": {
            "server": "172.20.188.236",
            "path": "/exports",
            "containerMountPath": "/nfs"
        }
    });
    let testNniManagerIp: any = JSON.stringify({
        "nniManagerIp": "0.0.0.0"
    });
    let adlTrainingService: AdlTrainingService;
    console.log(tmp.dirSync().name);

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
    });

    beforeEach(() => {
        adlTrainingService = component.get(AdlTrainingService);
        adlTrainingService.run()
    });

    afterEach(() => {
        adlTrainingService.cleanUp();
    });

    it('Set and get cluster metadata', async () => {
        await adlTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, testAdlTrialConfig2);
        await adlTrainingService.setClusterMetadata(TrialConfigMetadataKey.NNI_MANAGER_IP, testNniManagerIp);
        let data:string = await adlTrainingService.getClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG);
        chai.expect(data).to.be.equals(testAdlTrialConfig2);
    });

    it('Submit job', async () => {
        await adlTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, testAdlTrialConfig);
        // submit job
        const form: TrialJobApplicationForm = {
            sequenceId: 0,
            hyperParameters: {
                value: 'mock hyperparameters',
                index: 0
            }
        };
        const jobDetail: TrialJobDetail = await adlTrainingService.submitTrialJob(form);
        chai.expect(jobDetail.status).to.be.equals('WAITING');
        await adlTrainingService.cancelTrialJob(jobDetail.id);
        chai.expect(jobDetail.status).to.be.equals('USER_CANCELED');
    }).timeout(3000000);
});
