// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { assert, expect } from 'chai';
import request from 'request';

import { IocShim } from 'common/ioc_shim';
import { Database, DataStore } from '../../common/datastore';
import { ExperimentProfile, Manager } from '../../common/manager';
import { TrainingService } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { SqlDB } from '../../core/sqlDatabase';
import { MockedDataStore } from '../mock/datastore';
import { MockedTrainingService } from '../mock/trainingService';
import { RestServer, UnitTestHelpers } from 'rest_server';
import { MockedNNIManager } from '../mock/nniManager';
import { MockedExperimentManager } from '../mock/experimentManager';
import { TensorboardManager } from '../../common/tensorboardManager';
import { MockTensorboardManager } from '../mock/mockTensorboardManager';
import { UnitTestHelpers as ExpsMgrHelpers } from 'extensions/experiments_manager';
import globals from 'common/globals/unittest';
import { createRestHandler } from 'rest_server/restHandler';

let restServer: RestServer;

describe('Unit test for rest handler', () => {

    let ROOT_URL: string;

    before(async () => {
        ExpsMgrHelpers.setExperimentsManager(new MockedExperimentManager());
        prepareUnitTest();
        IocShim.clear();
        IocShim.bind(Database, SqlDB);
        IocShim.bind(DataStore, MockedDataStore);
        IocShim.bind(TrainingService, MockedTrainingService);
        IocShim.bind(Manager, MockedNNIManager);
        IocShim.bind(TensorboardManager, MockTensorboardManager);
        restServer = new RestServer(0, '');
        await restServer.start();
        const port = UnitTestHelpers.getPort(restServer);
        ROOT_URL = `http://localhost:${port}/api/v1/nni`;
        globals.rest.registerExpressRouter('/api/v1/nni', createRestHandler());
    });

    after(() => {
        restServer.shutdown();
        cleanupUnitTest();
    });

    it('Test GET check-status', (done: Mocha.Done) => {
        request.get(`${ROOT_URL}/check-status`, (err: Error, res: request.Response) => {
            if (err) {
                assert.fail(err.message);
            } else {
                expect(res.statusCode).to.equal(200);
            }
            done();
        });
    });

    it('Test GET trial-jobs/:id', (done: Mocha.Done) => {
        request.get(`${ROOT_URL}/trial-jobs/1234`, (err: Error, res: request.Response, body: any) => {
            if (err) {
                assert.fail(err.message);
            } else {
                expect(res.statusCode).to.equal(200);
                expect(JSON.parse(body).trialJobId).to.equal('1234');
            }
            done();
        });
    });

    it('Test GET experiment', (done: Mocha.Done) => {
        request.get(`${ROOT_URL}/experiment`, (err: Error, res: request.Response) => {
            if (err) {
                assert.fail(err.message);
            } else {
                expect(res.statusCode).to.equal(200);
            }
            done();
        });
    });

    it('Test GET trial-jobs', (done: Mocha.Done) => {
        request.get(`${ROOT_URL}/trial-jobs`, (err: Error, res: request.Response) => {
            expect(res.statusCode).to.equal(200);
            if (err) {
                assert.fail(err.message);
            }
            done();
        });
    });

    it('Test GET experiments-info', (done: Mocha.Done) => {
        request.get(`${ROOT_URL}/experiments-info`, (err: Error, res: request.Response) => {
            expect(res.statusCode).to.equal(200);
            if (err) {
                assert.fail(err.message);
            }
            done();
        });
    });

    it('Test change concurrent-trial-jobs', (done: Mocha.Done) => {
        request.get(`${ROOT_URL}/experiment`, (err: Error, res: request.Response, body: any) => {
            if (err) {
                assert.fail(err.message);
            } else {
                expect(res.statusCode).to.equal(200);
                const profile: ExperimentProfile = JSON.parse(body);
                if (profile.params && profile.params.trialConcurrency) {
                    profile.params.trialConcurrency = 10;
                }

                const req: request.Options = {
                    uri: `${ROOT_URL}/experiment?update_type=TRIAL_CONCURRENCY`,
                    method: 'PUT',
                    json: true,
                    body: profile
                };
                request(req, (error: Error, response: request.Response) => {
                    if (error) {
                        assert.fail(error.message);
                    } else {
                        expect(response.statusCode).to.equal(200);
                    }
                    done();
                });
            }
        });
    });
});
