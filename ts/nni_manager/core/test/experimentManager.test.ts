// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { assert, expect } from 'chai';
import * as fs from 'fs';
import { Container, Scope } from 'typescript-ioc';

import * as component from '../../common/component';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { ExpManager } from '../../common/expmanager';
import { ExperimentsManager } from '../experimentsManager';


describe('Unit test for experiment manager', function () {
    let expManager: ExperimentsManager;
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
        Container.bind(ExpManager).to(ExperimentsManager).scope(Scope.Singleton);
        expManager = component.get(ExperimentsManager);
        expManager.setExperimentPath('.experiment.test');
    });

    after(() => {
        if (fs.existsSync('.experiment.test')) {
            fs.unlinkSync('.experiment.test');
        }
        cleanupUnitTest();
    });

    it('test getExperimentsInfo', () => {
        return expManager.getExperimentsInfo().then(function (experimentsInfo: {[key: string]: any}) {
            expect(experimentsInfo['test']['status']).to.be.oneOf(['STOPPED', 'ERROR']);
        }).catch((error) => {
            assert.fail(error);
        })
    });
});
