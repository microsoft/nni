// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import chai from 'chai';
import chaiAsPromised from 'chai-as-promised';
import { cleanupUnitTest, prepareUnitTest } from '../../../common/utils';
import { WindowsCommands } from '../../../training_service/remote_machine/extends/windowsCommands';


describe('Unit Test for Windows Commands', () => {

    let windowsCommands: WindowsCommands

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
    });

    beforeEach(() => {
        windowsCommands = new WindowsCommands();
    });

    afterEach(() => {
    });

    it('joinPath', async () => {
        chai.expect(windowsCommands.joinPath("/root/", "\\first")).to.equal("\\root\\first");
        chai.expect(windowsCommands.joinPath("root/", "first")).to.equal("root\\first");
        chai.expect(windowsCommands.joinPath("\\root/", "\\first")).to.equal("\\root\\first");
        chai.expect(windowsCommands.joinPath("\\root\\", "\\first")).to.equal("\\root\\first");
        chai.expect(windowsCommands.joinPath("\\root", "first")).to.equal("\\root\\first");
        chai.expect(windowsCommands.joinPath("\\root\\", "first")).to.equal("\\root\\first");
        chai.expect(windowsCommands.joinPath("root\\", "first")).to.equal("root\\first");
        chai.expect(windowsCommands.joinPath("root\\")).to.equal("root\\");
        chai.expect(windowsCommands.joinPath("root")).to.equal("root");
        chai.expect(windowsCommands.joinPath(".\\root")).to.equal(".\\root");
        chai.expect(windowsCommands.joinPath("")).to.equal(".");
        chai.expect(windowsCommands.joinPath("..")).to.equal("..");
    })

    it('createFolder', async () => {
        chai.expect(windowsCommands.createFolder("test")).to.equal("mkdir \"test\"");
        chai.expect(windowsCommands.createFolder("test", true)).to.equal("mkdir \"test\"\r\nICACLS \"test\" /grant \"Users\":F");
    })

    it('allowPermission', async () => {
        chai.expect(windowsCommands.allowPermission(true, "test", "test1")).to.equal("ICACLS \"test\" /grant \"Users\":F /T\r\nICACLS \"test1\" /grant \"Users\":F /T\r\n");
        chai.expect(windowsCommands.allowPermission(false, "test")).to.equal("ICACLS \"test\" /grant \"Users\":F\r\n");
    })

    it('removeFolder', async () => {
        chai.expect(windowsCommands.removeFolder("test")).to.equal("rmdir /q \"test\"");
        chai.expect(windowsCommands.removeFolder("test", true)).to.equal("rmdir /s /q \"test\"");
        chai.expect(windowsCommands.removeFolder("test", true, false)).to.equal("rmdir /s \"test\"");
        chai.expect(windowsCommands.removeFolder("test", false, false)).to.equal("rmdir \"test\"");
        chai.expect(windowsCommands.removeFolder("test", true, true)).to.equal("rmdir /s /q \"test\"");
    })

    it('removeFiles', async () => {
        chai.expect(windowsCommands.removeFiles("test", "*.sh")).to.equal("del \"test\\*.sh\"");
        chai.expect(windowsCommands.removeFiles("test", "")).to.equal("del \"test\"");
    })

    it('readLastLines', async () => {
        chai.expect(windowsCommands.readLastLines("test", 3)).to.equal("powershell.exe Get-Content \"test\" -Tail 3");
    })

    it('isProcessAlive', async () => {
        chai.expect(windowsCommands.isProcessAliveCommand("test")).to.equal("powershell.exe Get-Process -Id (get-content \"test\") -ErrorAction SilentlyContinue");
        chai.expect(windowsCommands.isProcessAliveProcessOutput(
            {
                exitCode: 0,
                stdout: "",
                stderr: ""
            }
        )).to.equal(true);
        chai.expect(windowsCommands.isProcessAliveProcessOutput(
            {
                exitCode: 10,
                stdout: "",
                stderr: ""
            }
        )).to.equal(false);
    })

    it('extractFile', async () => {
        chai.expect(windowsCommands.extractFile("test.tar", "testfolder")).to.equal("tar -xf \"test.tar\" -C \"testfolder\"");
    })

    it('executeScript', async () => {
        chai.expect(windowsCommands.executeScript("test.sh", true)).to.equal("test.sh");
        chai.expect(windowsCommands.executeScript("test script'\"", false)).to.equal("test script'\"");
    })
});
