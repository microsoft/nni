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

`use strict`;

import * as path from 'path';
import * as chai from 'chai';
import * as fs from 'fs';
import * as tmp from 'tmp';
import * as chaiAsPromised from 'chai-as-promised';
import { prepareUnitTest, cleanupUnitTest, delay } from '../../common/utils';
import * as component from '../../common/component';
import { AetherTrainingService } from '../../training_service/aether/aetherTrainingService';
import { TrialConfigMetadataKey } from '../../training_service/common/trialConfigMetadataKey';
import { fstat } from 'fs';
import { TrialJobApplicationForm, TrialJobDetail } from '../../common/trainingService';
import { Deferred } from 'ts-deferred';

const aetherCodeDir: string = tmp.dirSync().name.split('\\').join('\\\\');
const aetherGraphPath: string = './training_service/test/graph.json';
const aetherGraphTmpPath: string = path.win32.join(aetherCodeDir, 'graph.json').split('\\').join('\\\\'); 
fs.copyFileSync(aetherGraphPath, aetherGraphTmpPath);

describe(`Unit Test for Aether Training Service`, () => {
    let trial_config = `{"command": "", "codeDir": "${aetherCodeDir}", "gpuNum": 0, "baseGraph": "${aetherGraphTmpPath}", "outputNodeAlias": "184eb95a", "outputName": "OutputFile"}`;
    let aetherTrainingService: AetherTrainingService;    
    const hparam = {
        'parameter_id': 0,
        'parameter_source': 'fake',
        'parameters': {
            'param': 'hello',
        }
    }

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
    })

    after(() => {
        cleanupUnitTest();
    })

    beforeEach(async () => {
        aetherTrainingService = component.get(AetherTrainingService);
        await aetherTrainingService.setClusterMetadata(TrialConfigMetadataKey.NNI_MANAGER_IP, `{"nniManagerIp": "127.0.0.1"}`); 
        aetherTrainingService.run();
    })

    afterEach(() => {
        aetherTrainingService.cleanUp();
    })

    it('Submit job and collect metric', async () => {
        await aetherTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, trial_config);
        const form: TrialJobApplicationForm = {
            jobType: 'TRIAL',
            hyperParameters: {
                   value: JSON.stringify(hparam),
                   index: 0,
            }
        };
        const jobDetail: TrialJobDetail = await aetherTrainingService.submitTrialJob(form);
        chai.expect(jobDetail.status).to.be.oneOf(['WAITING', 'RUNNING']);
        aetherTrainingService.listTrialJobs().then((jobList) => {
            chai.expect(jobList).to.be.lengthOf(1);
        });

        const metric_deferred = new Deferred<string>();
        const listener = function f1(metric: any) {
            chai.expect(metric.id).to.be.equals(jobDetail.id);
            metric_deferred.resolve(metric.data);
        };
        aetherTrainingService.addTrialJobMetricListener(listener);
        
        const metric_data: string = await metric_deferred.promise;
        aetherTrainingService.removeTrialJobMetricListener(listener);
    }).timeout(100000);

    it('Submit job and cancel', async () => {
        await aetherTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, trial_config);
        const form: TrialJobApplicationForm = {
            jobType: 'TRIAL',
            hyperParameters: {
                   value: JSON.stringify(hparam),
                   index: 0,
            }
        };
        const jobDetail: TrialJobDetail = await aetherTrainingService.submitTrialJob(form);
        chai.expect(jobDetail.status).to.be.oneOf(['WAITING', 'RUNNING']);

        await aetherTrainingService.cancelTrialJob(jobDetail.id);
        chai.expect(jobDetail.status).to.be.oneOf(['USER_CANCELED', 'SYS_CANCELED']);
    }).timeout(100000);
})