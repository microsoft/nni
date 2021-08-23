// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { assert, expect } from 'chai';
import * as fs from 'fs';
import { Container, Scope } from 'typescript-ioc';

import * as component from '../../common/component';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { ExperimentManager } from '../../common/experimentManager';
import { NNIExperimentsManager } from '../nniExperimentsManager';


describe('Unit test for experiment manager', function () {
    let experimentManager: NNIExperimentsManager;
    const mockedInfo = {
        "test": {
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

    before(() => {
        prepareUnitTest();
        fs.writeFileSync('.experiment.test', JSON.stringify(mockedInfo));
        Container.bind(ExperimentManager).to(NNIExperimentsManager).scope(Scope.Singleton);
        experimentManager = component.get(NNIExperimentsManager);
        experimentManager.setExperimentPath('.experiment.test');
    });

    after(() => {
        if (fs.existsSync('.experiment.test')) {
            fs.unlinkSync('.experiment.test');
        }
        cleanupUnitTest();
    });

    it('test getExperimentsInfo', () => {
        return experimentManager.getExperimentsInfo().then(function (experimentsInfo: {[key: string]: any}) {
            new Array(experimentsInfo)
            for (let idx in experimentsInfo) {
                if (experimentsInfo[idx]['id'] === 'test') {
                    expect(experimentsInfo[idx]['status']).to.be.oneOf(['STOPPED', 'ERROR']);
                    break;
                }
            }
        }).catch((error) => {
            assert.fail(error);
        })
    });
});
