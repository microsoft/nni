// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';
import * as fs from 'fs';
import * as tmp from 'tmp';
import * as component from '../../common/component';
import { TrialJobApplicationForm } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { PAIYarnTrainingService } from '../pai/paiYarn/paiYarnTrainingService';

// TODO: copy mockedTrail.py to local folder
const localCodeDir: string = tmp.dirSync().name
const mockedTrialPath: string = './training_service/test/mockedTrial.py'
fs.copyFileSync(mockedTrialPath, localCodeDir + '/mockedTrial.py')

describe('Unit Test for PAIYarnTrainingService', () => {
    let skip: boolean = false;
    let testPaiClusterInfo: any;
    let paiCluster: any;
    let paiTrialConfig : any;
    try {
        testPaiClusterInfo = JSON.parse(fs.readFileSync('../../.vscode/paiCluster.json', 'utf8'));
        paiCluster = `{\"userName\":\"${testPaiClusterInfo.userName}\",\"passWord\":\"${testPaiClusterInfo.passWord}\",\"host\":\"${testPaiClusterInfo.host}\"}`;
        paiTrialConfig = `{\"command\":\"echo hello && ls\",\"codeDir\":\"/tmp/nni/examples/trials/mnist",\"gpuNum\":\"1\",
\"cpuNum\":\"1\",\"memoryMB\":\"8196\",\"image\":\"openpai/pai.example.tensorflow\",\"dataDir\":\"\",\"outputDir\":\"\"}`;
    } catch (err) {
        console.log('Please configure rminfo.json to enable remote machine unit test.');
        skip = true;
    }

    let paiYarnTrainingService: PAIYarnTrainingService;

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
        paiYarnTrainingService = component.get(PAIYarnTrainingService);
        paiYarnTrainingService.run();
    });

    afterEach(() => {
        if (skip) {
            return;
        }
        paiYarnTrainingService.cleanUp();
    });

    it('Get PAI token', async () => {
        if (skip) {
            return;
        }
        console.log(`paiCluster is ${paiCluster}`)
        await paiYarnTrainingService.setClusterMetadata(TrialConfigMetadataKey.PAI_YARN_CLUSTER_CONFIG, paiCluster);
        await paiYarnTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, paiTrialConfig);
        const form: TrialJobApplicationForm = {
            sequenceId: 0,
            hyperParameters: { value: '', index: 0 }
        };
        try {
            const trialDetail = await paiYarnTrainingService.submitTrialJob(form);
            chai.expect(trialDetail.status).to.be.equals('WAITING');
        } catch(error) {
            console.log('Submit job failed:' + error);
            chai.assert(error)
        }
    });
});
