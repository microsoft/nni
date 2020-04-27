// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';
import * as component from '../../../common/component';
import { cleanupUnitTest, prepareUnitTest } from '../../../common/utils';
import { LinuxCommands } from '../extends/linuxCommands';
// import { TrialConfigMetadataKey } from '../trialConfigMetadataKey';


describe('Unit Test for linuxCommands', () => {

    let linuxCommands: LinuxCommands

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
    });

    beforeEach(() => {
        linuxCommands = component.get(LinuxCommands);
    });

    afterEach(() => {
    });

    it('join path', async () => {
        chai.expect(linuxCommands.joinPath("/root/", "/first")).to.equal("/root/first")
        chai.expect(linuxCommands.joinPath("/root", "first")).to.equal("/root/first")
        chai.expect(linuxCommands.joinPath("/root/", "first")).to.equal("/root/first")
        chai.expect(linuxCommands.joinPath("root/", "first")).to.equal("root/first")
        chai.expect(linuxCommands.joinPath("root/")).to.equal("root/")
        chai.expect(linuxCommands.joinPath("root")).to.equal("root")
        chai.expect(linuxCommands.joinPath("./root")).to.equal("./root")
        chai.expect(linuxCommands.joinPath("")).to.equal(".")
        chai.expect(linuxCommands.joinPath("..")).to.equal("..")
    })

    it('remove folder', async () => {
        chai.expect(linuxCommands.removeFolder("test")).to.equal("rm -df 'test'")
        chai.expect(linuxCommands.removeFolder("test", true)).to.equal("rm -rf 'test'");
        chai.expect(linuxCommands.removeFolder("test", true, false)).to.equal("rm -r 'test'");
        chai.expect(linuxCommands.removeFolder("test", false, false)).to.equal("rm 'test'");
    })

    it('create folder', async () => {
        chai.expect(linuxCommands.createFolder("test")).to.equal("mkdir -p 'test'")
        chai.expect(linuxCommands.createFolder("test", true)).to.equal("umask 0; mkdir -p 'test'")
    })
});
