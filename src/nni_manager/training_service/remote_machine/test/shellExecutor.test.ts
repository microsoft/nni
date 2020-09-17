// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';

import { ShellExecutor } from '../shellExecutor';
import { prepareUnitTest, cleanupUnitTest } from '../../../common/utils';

const LOCALFILE: string = 'localSshUTData';
const REMOTEFILE: string = 'remoteSshUTData';
const REMOTEFOLDER: string = 'remoteSshUTFolder';

async function copyFile(executor: ShellExecutor): Promise<void> {
    const remoteFullName = executor.joinPath(executor.getTempPath(), REMOTEFILE);
    await executor.copyFileToRemote(LOCALFILE, remoteFullName);
}

async function copyFileToRemoteLoop(executor: ShellExecutor): Promise<void> {
    const remoteFullName = executor.joinPath(executor.getTempPath(), REMOTEFILE);
    for (let i: number = 0; i < 3; i++) {
        await executor.copyFileToRemote(LOCALFILE, remoteFullName);
    }
}

async function getRemoteFileContentLoop(executor: ShellExecutor): Promise<void> {
    const remoteFullName = executor.joinPath(executor.getTempPath(), REMOTEFILE);
    for (let i: number = 0; i < 3; i++) {
        await executor.getRemoteFileContent(remoteFullName);
    }
}

describe('ShellExecutor test', () => {
    let skip: boolean = false;
    let isWindows: boolean;
    let rmMeta: any;
    try {
        rmMeta = JSON.parse(fs.readFileSync('../../.vscode/rminfo.json', 'utf8'));
        console.log(rmMeta);
    } catch (err) {
        console.log(`Please configure rminfo.json to enable remote machine test. ${err}`);
        skip = true;
    }

    before(async () => {
        chai.should();
        chai.use(chaiAsPromised);
        if (!fs.existsSync(LOCALFILE)){
            await cpp.exec(`echo '1234' > ${LOCALFILE}`);
        }
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
        fs.unlinkSync(LOCALFILE);
    });

    it('Test mkdir', async () => {
        if (skip) {
            return;
        }
        const executor: ShellExecutor = new ShellExecutor();
        await executor.initialize(rmMeta);
        const remoteFullPath = executor.joinPath(executor.getTempPath(), REMOTEFOLDER);
        let result = await executor.createFolder(remoteFullPath, false);
        chai.expect(result).eq(true);
        const commandResult = await executor.executeScript("dir");
        chai.expect(commandResult.exitCode).eq(0);
        result = await executor.removeFolder(remoteFullPath);
        chai.expect(result).eq(true);
        await executor.close();
    });

    it('Test ShellExecutor', async () => {
        if (skip) {
            return;
        }
        const executor: ShellExecutor = new ShellExecutor();
        await executor.initialize(rmMeta);
        await copyFile(executor);
        await copyFileToRemoteLoop(executor);
        await getRemoteFileContentLoop(executor);
        await executor.close();
    });

    it('Test preCommand-1', async () => {
        if (skip) {
            return;
        }
        const executor: ShellExecutor = new ShellExecutor();
        await executor.initialize(rmMeta);
        const result = await executor.executeScript("ver", false, false);
        isWindows = result.exitCode == 0 && result.stdout.search("Windows") > -1;
        await executor.close();
    });

    it('Test preCommand-2', async () => {
        if (skip) {
            return;
        }
        const executor: ShellExecutor = new ShellExecutor();
        rmMeta.preCommand = isWindows ? "set TEST_PRE_COMMAND=test_pre_command" : "export TEST_PRE_COMMAND=test_pre_command";
        await executor.initialize(rmMeta);
        const command = isWindows ? "python -c \"import os; print(os.environ.get(\'TEST_PRE_COMMAND\'))\"" : "python3 -c \"import os; print(os.environ.get(\'TEST_PRE_COMMAND\'))\"";
        const result = (await executor.executeScript(command, false, false)).stdout.replace(/[\ +\r\n]/g, "");
        chai.expect(result).eq("test_pre_command");
        await executor.close();
    });
});
