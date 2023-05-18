// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { assert, expect } from 'chai';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { ExperimentsManager } from '../../extensions/experiments_manager';
import globals from '../../common/globals/unittest';

let tempDir: string | null = null;
let experimentManager: ExperimentsManager;
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

describe('Unit test for experiment manager', function () {
    before(() => {
        prepareUnitTest();
        tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nni-ut-'));
        globals.paths.experimentsList = path.join(tempDir, '.experiment');
        fs.writeFileSync(globals.paths.experimentsList, JSON.stringify(mockedInfo));
        experimentManager = new ExperimentsManager();
    });

    after(() => {
        if (tempDir !== null) {
            fs.rmSync(tempDir, { force: true, recursive: true });
        }
        cleanupUnitTest();
    });

    it('test getExperimentsInfo', async () => {
        const experimentsInfo: {[key: string]: any} = await experimentManager.getExperimentsInfo();
        for (let idx in experimentsInfo) {
            if (experimentsInfo[idx]['id'] === 'test') {
                expect(experimentsInfo[idx]['status']).to.be.oneOf(['STOPPED', 'ERROR']);
                break;
            }
        }
    });
});
