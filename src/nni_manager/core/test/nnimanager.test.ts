// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as os from 'os';
import { assert, expect } from 'chai';
import { Container, Scope } from 'typescript-ioc';

import * as component from '../../common/component';
import { Database, DataStore } from '../../common/datastore';
import { Manager, ExperimentProfile} from '../../common/manager';
import { TrainingService } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { NNIDataStore } from '../nniDataStore';
import { NNIManager } from '../nnimanager';
import { SqlDB } from '../sqlDatabase';
import { MockedTrainingService } from './mockedTrainingService';
import { MockedDataStore } from './mockedDatastore';
import * as path from 'path';

async function initContainer(): Promise<void> {
    prepareUnitTest();
    Container.bind(TrainingService).to(MockedTrainingService).scope(Scope.Singleton);
    Container.bind(Manager).to(NNIManager).scope(Scope.Singleton);
    Container.bind(Database).to(SqlDB).scope(Scope.Singleton);
    Container.bind(DataStore).to(MockedDataStore).scope(Scope.Singleton);
    await component.get<DataStore>(DataStore).init();
}

describe('Unit test for nnimanager', function () {
    this.timeout(10000);

    let nniManager: Manager;

    let ClusterMetadataKey = 'mockedMetadataKey';

    let experimentParams = {
        authorName: 'zql',
        experimentName: 'naive_experiment',
        trialConcurrency: 3,
        maxExecDuration: 5,
        maxTrialNum: 3,
        trainingServicePlatform: 'local',
        searchSpace: '{"lr": {"_type": "choice", "_value": [0.01,0.001]}}',
        tuner: {
            builtinTunerName: 'TPE',
            classArgs: {
                optimize_mode: 'maximize'
            },
            checkpointDir: '',
        },
        assessor: {
            builtinAssessorName: 'Medianstop',
            checkpointDir: '',
        }
    }

    let updateExperimentParams = {
        authorName: '',
        experimentName: 'another_experiment',
        trialConcurrency: 2,
        maxExecDuration: 6,
        maxTrialNum: 2,
        trainingServicePlatform: 'local',
        searchSpace: '{"lr": {"_type": "choice", "_value": [0.01,0.001]}}',
        tuner: {
            builtinTunerName: 'TPE',
            classArgs: {
                optimize_mode: 'maximize'
            },
            checkpointDir: '',
            gpuNum: 0
        },
        assessor: {
            builtinAssessorName: 'Medianstop',
            checkpointDir: '',
            gpuNum: 1
        }
    }

    let experimentProfile = {
        params: updateExperimentParams,
        id: 'test',
        execDuration: 0,
        nextSequenceId: 0,
        revision: 0
    }


    before(async () => {
        await initContainer();
        nniManager = component.get(Manager);
        const expId: string = await nniManager.startExperiment(experimentParams);
        assert.strictEqual(expId, 'unittest');
    })

    after(async () => {
        await setTimeout(() => {nniManager.stopExperiment()},15000);
        cleanupUnitTest();
    })



    it('test addCustomizedTrialJob', () => {
        return nniManager.addCustomizedTrialJob('"hyperParams"').then(() => {

        }).catch((error) => {
            assert.fail(error);
        })
    })


    it('test listTrialJobs', () => {
        return nniManager.listTrialJobs().then(function (trialjobdetails) {
            expect(trialjobdetails.length).to.be.equal(2);
        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test getTrialJob valid', () => {
        //query a exist id
        return nniManager.getTrialJob('1234').then(function (trialJobDetail) {
            expect(trialJobDetail.id).to.be.equal('1234');
        }).catch((error) => {
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

    it('test cancelTrialJobByUser', () => {
        return nniManager.cancelTrialJobByUser('1234').then(() => {

        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test getExperimentProfile', () => {
        return nniManager.getExperimentProfile().then((experimentProfile) => {
            expect(experimentProfile.id).to.be.equal('unittest');
            expect(experimentProfile.logDir).to.be.equal(path.join(os.homedir(),'nni','experiments','unittest'));

        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test updateExperimentProfile TRIAL_CONCURRENCY',  () => {
        return nniManager.updateExperimentProfile(experimentProfile, 'TRIAL_CONCURRENCY').then(() => {
            nniManager.getExperimentProfile().then((updateProfile) => {
                expect(updateProfile.params.trialConcurrency).to.be.equal(2);
            });
        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test updateExperimentProfile MAX_EXEC_DURATION',  () => {
        return nniManager.updateExperimentProfile(experimentProfile, 'MAX_EXEC_DURATION').then(() => {
            nniManager.getExperimentProfile().then((updateProfile) => {
                expect(updateProfile.params.maxExecDuration).to.be.equal(6);
            });
        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test updateExperimentProfile SEARCH_SPACE',  () => {
        return nniManager.updateExperimentProfile(experimentProfile, 'SEARCH_SPACE').then(() => {
            nniManager.getExperimentProfile().then((updateProfile) => {
                expect(updateProfile.params.searchSpace).to.be.equal('{"lr": {"_type": "choice", "_value": [0.01,0.001]}}');
            });
        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test updateExperimentProfile MAX_TRIAL_NUM',  () => {
        return nniManager.updateExperimentProfile(experimentProfile, 'MAX_TRIAL_NUM').then(() => {
            nniManager.getExperimentProfile().then((updateProfile) => {
                expect(updateProfile.params.maxTrialNum).to.be.equal(2);
            });
        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test getStatus', () => {
        assert.strictEqual(nniManager.getStatus().status,'RUNNING');
    })

    it('test getMetricData with trialJobId', () => {
        //query a exist trialJobId
        return nniManager.getMetricData('4321', 'CUSTOM').then((metricData) => {
            expect(metricData.length).to.be.equal(1);
            expect(metricData[0].trialJobId).to.be.equal('4321');
            expect(metricData[0].parameterId).to.be.equal('param1');
        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test getMetricData with invalid trialJobId', () => {
        //query an invalid trialJobId
        return nniManager.getMetricData('43210', 'CUSTOM').then((metricData) => {
            assert.fail();
        }).catch((error) => {
        })
    })

    it('test getTrialJobStatistics', () => {
        // get 3 trial jobs (init, addCustomizedTrialJob, cancelTrialJobByUser)
        return nniManager.getTrialJobStatistics().then(function (trialJobStatistics) {
            expect(trialJobStatistics.length).to.be.equal(2);
            if (trialJobStatistics[0].trialJobStatus === 'WAITING') {
                expect(trialJobStatistics[0].trialJobNumber).to.be.equal(2);
                expect(trialJobStatistics[1].trialJobNumber).to.be.equal(1);
            }
            else {
                expect(trialJobStatistics[1].trialJobNumber).to.be.equal(2);
                expect(trialJobStatistics[0].trialJobNumber).to.be.equal(1);
            }
        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test addCustomizedTrialJob reach maxTrialNum', () => {
        // test currSubmittedTrialNum reach maxTrialNum
        return nniManager.addCustomizedTrialJob('"hyperParam"').then(() => {
            nniManager.getTrialJobStatistics().then(function (trialJobStatistics) {
                if (trialJobStatistics[0].trialJobStatus === 'WAITING')
                    expect(trialJobStatistics[0].trialJobNumber).to.be.equal(2);
                else
                    expect(trialJobStatistics[1].trialJobNumber).to.be.equal(2);
            })
        }).catch((error) => {
            assert.fail(error);
        })
    })

    it('test resumeExperiment', async () => {
       //TODO: add resume experiment unit test
    })

})
