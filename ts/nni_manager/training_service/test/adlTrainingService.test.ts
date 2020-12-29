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

const localCodeDir: string = tmp.dirSync().name

describe('Unit Test for AdlTrainingService', () => {
    let skip: boolean = false;
    try {
        const testKubeflowConfig = fs.readFileSync('/home/vsts/.kube/config', 'utf8');
    } catch (err) {
        console.log('Please have kubernetes cluster to enable its training service unit test.');
        skip = true;
    }

    let testAdlTrialConfig: any = JSON.stringify({
        "command": "python3 /root/apps/nni_linear_regression/main.py",
        "codeDir": ".",
        "gpuNum": 0,
        "image": "test.image:latest",
        "imagePullSecrets": [
            {
                "name": "stagingsecrets"
            }
        ],
        "nfs": {
            "server": "172.20.188.236",
            "path": "/exports",
            "containerMountPath": "/nfs"
        },
        "memorySize": "1Gi",
        "cpuNum": 1
    });
    let testAdlTrialConfig2: any = JSON.stringify({
        "command": "python3 /root/apps/nni_linear_regression/main.py",
        "codeDir": ".",
        "gpuNum": 0,
        "image": "test.image:latest",
        "imagePullSecrets": [
            {
                "name": "stagingsecrets"
            }
        ],
        "adaptive": true,
        "checkpoint": {
            "storageClass": "aws-efs",
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
        if (skip) {
            return;
        }
        adlTrainingService = component.get(AdlTrainingService);
        adlTrainingService.run()
    });

    afterEach(() => {
        if (skip) {
            return;
        }
        adlTrainingService.cleanUp();
    });

    it('Set and get cluster metadata', async () => {
        if (skip) {
            return;
        }
        await adlTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, testAdlTrialConfig2);
        await adlTrainingService.setClusterMetadata(TrialConfigMetadataKey.NNI_MANAGER_IP, testNniManagerIp);
        let data:string = await adlTrainingService.getClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG);
        chai.expect(data).to.be.equals(testAdlTrialConfig2);
    });

    it('Submit job', async () => {
        if (skip) {
            return;
        }
        // job without given checkpoint, with resource config
        await adlTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, testAdlTrialConfig);
        let form: TrialJobApplicationForm = {
            sequenceId: 0,
            hyperParameters: {
                value: 'mock hyperparameters',
                index: 0
            }
        };
        let jobDetail: TrialJobDetail = await adlTrainingService.submitTrialJob(form);
        chai.expect(jobDetail.status).to.be.equals('WAITING');
        await adlTrainingService.cancelTrialJob(jobDetail.id);
        chai.expect(jobDetail.status).to.be.equals('USER_CANCELED');
        // job with given checkpoint
        await adlTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, testAdlTrialConfig2);
        form = {
            sequenceId: 0,
            hyperParameters: {
                value: 'mock hyperparameters',
                index: 0
            }
        };
        jobDetail = await adlTrainingService.submitTrialJob(form);
        chai.expect(jobDetail.status).to.be.equals('WAITING');
        await adlTrainingService.cancelTrialJob(jobDetail.id);
        chai.expect(jobDetail.status).to.be.equals('USER_CANCELED');
    }).timeout(3000000);
});
