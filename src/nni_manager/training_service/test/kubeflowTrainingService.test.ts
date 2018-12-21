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
import { KubeflowTrainingService } from '../kubeflow/kubeflowTrainingService';

// TODO: copy mockedTrail.py to local folder
const localCodeDir: string = tmp.dirSync().name
const mockedTrialPath: string = './training_service/test/mockedTrial.py'
fs.copyFileSync(mockedTrialPath, localCodeDir + '/mockedTrial.py')

describe('Unit Test for KubeflowTrainingService', () => {
    let skip: boolean = false;
    let testKubeflowConfig: any;
    let testKubeflowTrialConfig : any;
    try {
        testKubeflowConfig = JSON.parse(fs.readFileSync('../../.vscode/kubeflowCluster.json', 'utf8'));
        testKubeflowTrialConfig = `{\"command\":\"python3 mnist.py\",\"codeDir\":\"/tmp/nni/examples/trials/mnist",\"gpuNum\":\"1\",\"cpuNum\":\"2\",\"memoryMB\":\"8196\",\"image\":\"msranni/nni:latest\"}`;
    } catch (err) {
        console.log('Please configure kubeflowCluster.json to enable kubeflow training service unit test.');
        skip = true;
    }

    let kubeflowTrainingService: KubeflowTrainingService;

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
        kubeflowTrainingService = component.get(KubeflowTrainingService);        
    });

    afterEach(() => {
        if (skip) {
            return;
        }
        kubeflowTrainingService.cleanUp();
    });

    it('Set cluster metadata', async () => {
        if (skip) {
            return;
        }
        await kubeflowTrainingService.setClusterMetadata(TrialConfigMetadataKey.KUBEFLOW_CLUSTER_CONFIG, testKubeflowConfig),
        await kubeflowTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, testKubeflowTrialConfig);                
    });
});