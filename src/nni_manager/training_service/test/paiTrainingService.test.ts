/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';
import * as fs from 'fs';
import * as tmp from 'tmp';
import * as component from '../../common/component';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { PAITrainingService } from '../pai/paiTrainingService';

// TODO: copy mockedTrail.py to local folder
const localCodeDir: string = tmp.dirSync().name
const mockedTrialPath: string = './training_service/test/mockedTrial.py'
fs.copyFileSync(mockedTrialPath, localCodeDir + '/mockedTrial.py')

describe('Unit Test for PAITrainingService', () => {
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

    let paiTrainingService: PAITrainingService;

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
        paiTrainingService = component.get(PAITrainingService);
        paiTrainingService.run();
    });

    afterEach(() => {
        if (skip) {
            return;
        }
        paiTrainingService.cleanUp();
    });

    it('Get PAI token', async () => {
        if (skip) {
            return;
        }
        console.log(`paiCluster is ${paiCluster}`)
        await paiTrainingService.setClusterMetadata(TrialConfigMetadataKey.PAI_CLUSTER_CONFIG, paiCluster);
        await paiTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, paiTrialConfig);
        try {
            const trialDetail = await paiTrainingService.submitTrialJob({jobType : 'TRIAL'});
            chai.expect(trialDetail.status).to.be.equals('WAITING');
        } catch(error) {
            console.log('Submit job failed:' + error);
            chai.assert(error)            
        }
    });
});