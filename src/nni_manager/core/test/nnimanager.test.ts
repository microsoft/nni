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

import { assert, expect } from 'chai';
import { Container, Scope } from 'typescript-ioc';

import * as component from '../../common/component';
import { Database, DataStore } from '../../common/datastore';
import { Manager } from '../../common/manager';
import { TrainingService } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { NNIDataStore } from '../nniDataStore';
import { NNIManager } from '../nnimanager';
import { SqlDB } from '../sqlDatabase';
import { MockedTrainingService } from './mockedTrainingService';

async function initContainer(): Promise<void> {
    prepareUnitTest();
    Container.bind(TrainingService).to(MockedTrainingService).scope(Scope.Singleton);
    Container.bind(Manager).to(NNIManager).scope(Scope.Singleton);
    Container.bind(Database).to(SqlDB).scope(Scope.Singleton);
    Container.bind(DataStore).to(NNIDataStore).scope(Scope.Singleton);
    await component.get<DataStore>(DataStore).init();
}

describe('Unit test for nnimanager', function () {
    this.timeout(10000);

    let nniManager: Manager;

    let ClusterMetadataKey = 'mockedMetadataKey';

    let experimentParams = {
        authorName: 'zql',
        experimentName: 'naive_experiment',
        trialConcurrency: 2,
        maxExecDuration: 5,
        maxTrialNum: 2,
        trainingServicePlatform: 'local',
        searchSpace: '{"x":1}',
        tuner: {
            className: 'TPE',
            classArgs: {
                optimize_mode: 'maximize'
            },
            checkpointDir: '',
            gpuNum: 0
        },
        assessor: {
            className: 'Medianstop',
            checkpointDir: '',
            gpuNum: 1
        }
    }

    before(async () => {
        await initContainer();
        nniManager = component.get(Manager);
        const expId: string = await nniManager.startExperiment(experimentParams);
        assert(expId);
    });

    after(async () => {
        await nniManager.stopExperiment();
        cleanupUnitTest();
    })

    it('test resumeExperiment', () => {
        //TODO: add resume experiment unit test
    })

    it('test listTrialJobs', () => {
        //FIXME: not implemented
        //return nniManager.listTrialJobs().then(function (trialJobDetails) {
        //    expect(trialJobDetails.length).to.be.equal(2);
        //}).catch(function (error) {
        //    assert.fail(error);
        //})
    })

    it('test getTrialJob valid', () => {
        //query a exist id
        return nniManager.getTrialJob('1234').then(function (trialJobDetail) {
            expect(trialJobDetail.id).to.be.equal('1234');
        }).catch(function (error) {
            assert.fail(error);
        })
    })

    it('test getTrialJob with invalid id', () => {
        //query a not exist id, and the function should throw error, and should not process then() method
        return nniManager.getTrialJob('4567').then((jobid) => {
            assert.fail();
        }).catch((error) => {
            assert.isTrue(true);
        })
    })

    it('test getClusterMetadata', () => {
        //default value is "default"
        return nniManager.getClusterMetadata(ClusterMetadataKey).then(function (value) {
            expect(value).to.equal("default");
        });
    })

    it('test setClusterMetadata and getClusterMetadata', () => {
        //set a valid key
        return nniManager.setClusterMetadata(ClusterMetadataKey, "newdata").then(() => {
            return nniManager.getClusterMetadata(ClusterMetadataKey).then(function (value) {
                expect(value).to.equal("newdata");
            });
        }).catch((error) => {
            console.log(error);
        })
    })

    //TODO: complete ut
    it('test cancelTrialJobByUser', () => {
        return nniManager.cancelTrialJobByUser('1234').then(() => {

        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test addCustomizedTrialJob', () => {
        return nniManager.addCustomizedTrialJob('hyperParams').then(() => {

        }).catch((error) => {
            assert.fail(error);
        })
    })
})
