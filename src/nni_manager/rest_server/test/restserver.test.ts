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

// tslint:disable-next-line:no-implicit-dependencies
import { assert, expect } from 'chai';
// tslint:disable-next-line:no-implicit-dependencies
import * as request from 'request';
import { Container } from 'typescript-ioc';

import * as component from '../../common/component';
import { DataStore } from '../../common/datastore';
import { ExperimentProfile, Manager } from '../../common/manager';
import { TrainingService } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { MockedDataStore } from '../../core/test/mockedDatastore';
import { MockedTrainingService } from '../../core/test/mockedTrainingService';
import { NNIRestServer } from '../nniRestServer';
import { testManagerProvider } from './mockedNNIManager';

describe('Unit test for rest server', () => {

    let ROOT_URL: string;

    before((done: Mocha.Done) => {
        prepareUnitTest();
        Container.bind(Manager).provider(testManagerProvider);
        Container.bind(DataStore).to(MockedDataStore);
        Container.bind(TrainingService).to(MockedTrainingService);
        const restServer: NNIRestServer = component.get(NNIRestServer);
        restServer.start().then(() => {
            ROOT_URL = `${restServer.endPoint}/api/v1/nni`;
            done();
        }).catch((e: Error) => {
            assert.fail(`Failed to start rest server: ${e.message}`);
        });
    });

    after(() => {
        component.get<NNIRestServer>(NNIRestServer).stop();
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
        // tslint:disable-next-line:no-any
        request.get(`${ROOT_URL}/trial-jobs/1234`, (err: Error, res: request.Response, body: any) => {
            if (err) {
                assert.fail(err.message);
            } else {
                expect(res.statusCode).to.equal(200);
                expect(JSON.parse(body).id).to.equal('1234');
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

    it('Test change concurrent-trial-jobs', (done: Mocha.Done) => {
        // tslint:disable-next-line:no-any
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

    it('Test PUT experiment/cluster-metadata bad key', (done: Mocha.Done) => {
        const req: request.Options = {
            uri: `${ROOT_URL}/experiment/cluster-metadata`,
            method: 'PUT',
            json: true,
            body: {
                exception_test_key: 'test'
            }
        };
        request(req, (err: Error, res: request.Response) => {
            if (err) {
                assert.fail(err.message);
            } else {
                expect(res.statusCode).to.equal(400);
            }
            done();
        });
    });

    it('Test PUT experiment/cluster-metadata', (done: Mocha.Done) => {
        const req: request.Options = {
            uri: `${ROOT_URL}/experiment/cluster-metadata`,
            method: 'PUT',
            json: true,
            body: {
                machine_list: [{
                    ip: '10.10.10.101',
                    port: 22,
                    username: 'test',
                    passwd: '1234'
                }, {
                    ip: '10.10.10.102',
                    port: 22,
                    username: 'test',
                    passwd: '1234'
                }]
            }
        };
        request(req, (err: Error, res: request.Response) => {
            if (err) {
                assert.fail(err.message);
            } else {
                expect(res.statusCode).to.equal(200);
            }
            done();
        });
    });
});
