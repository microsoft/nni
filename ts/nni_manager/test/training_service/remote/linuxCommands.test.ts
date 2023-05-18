// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import chai from 'chai';
import chaiAsPromised from 'chai-as-promised';
import { cleanupUnitTest, prepareUnitTest } from '../../../common/utils';
import { LinuxCommands } from '../../../training_service/remote_machine/extends/linuxCommands';


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
        linuxCommands = new LinuxCommands();
    });

    afterEach(() => {
    });

    it('joinPath', async () => {
        chai.expect(linuxCommands.joinPath("/root/", "/first")).to.equal("/root/first");
        chai.expect(linuxCommands.joinPath("/root", "first")).to.equal("/root/first");
        chai.expect(linuxCommands.joinPath("/root/", "first")).to.equal("/root/first");
        chai.expect(linuxCommands.joinPath("root/", "first")).to.equal("root/first");
        chai.expect(linuxCommands.joinPath("root/")).to.equal("root/");
        chai.expect(linuxCommands.joinPath("root")).to.equal("root");
        chai.expect(linuxCommands.joinPath("./root")).to.equal("./root");
        chai.expect(linuxCommands.joinPath("")).to.equal(".");
        chai.expect(linuxCommands.joinPath("..")).to.equal("..");
    })

    it('createFolder', async () => {
        chai.expect(linuxCommands.createFolder("test")).to.equal("mkdir -p 'test'");
        chai.expect(linuxCommands.createFolder("test", true)).to.equal("umask 0; mkdir -p 'test'");
    })

    it('allowPermission', async () => {
        chai.expect(linuxCommands.allowPermission(true, "test", "test1")).to.equal("chmod 777 -R 'test' 'test1'");
        chai.expect(linuxCommands.allowPermission(false, "test")).to.equal("chmod 777 'test'");
    })

    it('removeFolder', async () => {
        chai.expect(linuxCommands.removeFolder("test")).to.equal("rm -df 'test'");
        chai.expect(linuxCommands.removeFolder("test", true)).to.equal("rm -rf 'test'");
        chai.expect(linuxCommands.removeFolder("test", true, false)).to.equal("rm -r 'test'");
        chai.expect(linuxCommands.removeFolder("test", false, false)).to.equal("rm 'test'");
    })

    it('removeFiles', async () => {
        chai.expect(linuxCommands.removeFiles("test", "*.sh")).to.equal("rm 'test/*.sh'");
        chai.expect(linuxCommands.removeFiles("test", "")).to.equal("rm 'test'");
    })

    it('readLastLines', async () => {
        chai.expect(linuxCommands.readLastLines("test", 3)).to.equal("tail -n 3 'test'");
    })

    it('isProcessAlive', async () => {
        chai.expect(linuxCommands.isProcessAliveCommand("test")).to.equal("kill -0 `cat 'test'`");
        chai.expect(linuxCommands.isProcessAliveProcessOutput(
            {
                exitCode: 0,
                stdout: "",
                stderr: ""
            }
        )).to.equal(true);
        chai.expect(linuxCommands.isProcessAliveProcessOutput(
            {
                exitCode: 10,
                stdout: "",
                stderr: ""
            }
        )).to.equal(false);
    })

    it('extractFile', async () => {
        chai.expect(linuxCommands.extractFile("test.tar", "testfolder")).to.equal("tar -oxzf 'test.tar' -C 'testfolder'");
    })

    it('executeScript', async () => {
        chai.expect(linuxCommands.executeScript("test.sh", true)).to.equal("bash 'test.sh'");
        chai.expect(linuxCommands.executeScript("test script'\"", false)).to.equal(`bash -c \"test script'\\""`);
    })
});
