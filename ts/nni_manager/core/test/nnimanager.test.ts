// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as os from 'os';
import { assert, expect } from 'chai';
import { Container, Scope } from 'typescript-ioc';

import * as component from '../../common/component';
import { Database, DataStore } from '../../common/datastore';
import { Manager, ExperimentProfile} from '../../common/manager';
import { ExperimentManager } from '../../common/experimentManager';
import { TrainingService } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { NNIExperimentsManager } from '../nniExperimentsManager';
import { NNIManager } from '../nnimanager';
import { SqlDB } from '../sqlDatabase';
import { MockedTrainingService } from './mockedTrainingService';
import { MockedDataStore } from './mockedDatastore';
import * as path from 'path';

async function initContainer(): Promise<void> {
    prepareUnitTest();
    Container.bind(Manager).to(NNIManager).scope(Scope.Singleton);
    Container.bind(Database).to(SqlDB).scope(Scope.Singleton);
    Container.bind(DataStore).to(MockedDataStore).scope(Scope.Singleton);
    Container.bind(ExperimentManager).to(NNIExperimentsManager).scope(Scope.Singleton);
    await component.get<DataStore>(DataStore).init();
}

describe('Unit test for nnimanager', function () {
    this.timeout(10000);

    let nniManager: NNIManager;

    let ClusterMetadataKey = 'mockedMetadataKey';

    let experimentParams = {
        experimentName: 'naive_experiment',
        trialConcurrency: 3,
        maxExperimentDuration: '5s',
        maxTrialNumber: 3,
        trainingService: {
            platform: 'local'
        },
        searchSpace: {'lr': {'_type': 'choice', '_value': [0.01,0.001]}},
        tuner: {
            name: 'TPE',
            classArgs: {
                optimize_mode: 'maximize'
            }
        },
        assessor: {
            name: 'Medianstop'
        },
        trialCommand: 'echo hello',
        trialCodeDirectory: '',
        debug: true
    }

    let updateExperimentParams = {
        experimentName: 'another_experiment',
        trialConcurrency: 2,
        maxExperimentDuration: '6s',
        maxTrialNumber: 2,
        trainingService: {
            platform: 'local'
        },
        searchSpace: '{"lr": {"_type": "choice", "_value": [0.01,0.001]}}',
        tuner: {
            name: 'TPE',
            classArgs: {
                optimize_mode: 'maximize'
            }
        },
        assessor: {
            name: 'Medianstop'
        },
        trialCommand: 'echo hello',
        trialCodeDirectory: '',
        debug: true
    }

    let experimentProfile = {
        params: updateExperimentParams,
        id: 'test',
        execDuration: 0,
        logDir: '',
        startTime: 0,
        nextSequenceId: 0,
        revision: 0
    }

    let mockedInfo = {
        "unittest": {
            "port": 8080,
            "startTime": 1605246730756,
            "endTime": "N/A",
            "status": "INITIALIZED",
            "platform": "local",
            "experimentName": "testExp",
            "tag": [], "pid": 11111,
            "webuiUrl": [],
            "logDir": null
        }
    }


    before(async () => {
        await initContainer();
        fs.writeFileSync('.experiment.test', JSON.stringify(mockedInfo));
        const experimentsManager: ExperimentManager = component.get(ExperimentManager);
        experimentsManager.setExperimentPath('.experiment.test');
        nniManager = component.get(Manager);

        const expId: string = await nniManager.startExperiment(experimentParams);
        assert.strictEqual(expId, 'unittest');

        // TODO:
        // In current architecture we cannot prevent NNI manager from creating a training service.
        // The training service must be manually stopped here or its callbacks will block exit.
        // I'm planning on a custom training service register system similar to custom tuner,
        // and when that is done we can let NNI manager to use MockedTrainingService through config.
        const manager = nniManager as any;
        manager.trainingService.removeTrialJobMetricListener(manager.trialJobMetricListener);
        manager.trainingService.cleanUp();

        manager.trainingService = new MockedTrainingService();
        manager.stopExperiment = () => manager.trainingService.cleanUp();
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
            expect(trialJobDetail.trialJobId).to.be.equal('1234');
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

    it('test cancelTrialJobByUser', () => {
        return nniManager.cancelTrialJobByUser('1234').then(() => {

        }).catch((error) => {
            console.log(error);
            assert.fail(error);
        })
    })

    it('test getExperimentProfile', () => {
        return nniManager.getExperimentProfile().then((experimentProfile) => {
            expect(experimentProfile.id).to.be.equal('unittest');
            expect(experimentProfile.logDir).to.be.equal(path.join(os.homedir(),'nni-experiments','unittest'));

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
                expect(updateProfile.params.maxExperimentDuration).to.be.equal('6s');
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
                expect(updateProfile.params.maxTrialNumber).to.be.equal(2);
            });
        }).catch((error: any) => {
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

    it('test addCustomizedTrialJob reach maxTrialNumber', () => {
        // test currSubmittedTrialNum reach maxTrialNumber
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
