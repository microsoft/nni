// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import * as chai from 'chai';
import { cleanupUnitTest, prepareUnitTest } from '../../../common/utils';
import chaiAsPromised = require("chai-as-promised");
import { AMLClient } from '../aml/amlClient';


describe('Unit Test for amlClient', () => {

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
    });

    it('test parseContent', async () => {

        let amlClient: AMLClient = new AMLClient('', '', '', '', '', '', '', '');
    
        chai.assert.equal(amlClient.parseContent('test', 'test:1234'), '1234', "The content should be 1234");
        chai.assert.equal(amlClient.parseContent('test', 'abcd:1234'), '', "The content should be null");
    });
});
