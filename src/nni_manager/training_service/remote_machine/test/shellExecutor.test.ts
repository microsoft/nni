// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';

import { Client } from 'ssh2';
import { ShellExecutor } from '../shellExecutor';
import { prepareUnitTest, cleanupUnitTest } from '../../../common/utils';

const LOCALFILE: string = '/tmp/localSshclientUTData';
const REMOTEFILE: string = '/tmp/remoteSshclientUTData';
const REMOTEFOLDER: string = '/tmp/remoteSshclientUTFolder';

async function copyFile(executor: ShellExecutor): Promise<void> {
    await executor.copyFileToRemote(LOCALFILE, REMOTEFILE);
}

async function copyFileToRemoteLoop(executor: ShellExecutor): Promise<void> {
    for (let i: number = 0; i < 10; i++) {
        // console.log(i);
        await executor.copyFileToRemote(LOCALFILE, REMOTEFILE);
    }
}

async function getRemoteFileContentLoop(executor: ShellExecutor): Promise<void> {
    for (let i: number = 0; i < 10; i++) {
        // console.log(i);
        await executor.getRemoteFileContent(REMOTEFILE);
    }
}

describe('ShellExecutor test', () => {
    let skip: boolean = false;
    let rmMeta: any;
    try {
        rmMeta = JSON.parse(fs.readFileSync('../../.vscode/rminfo.json', 'utf8'));
        console.log(rmMeta);
    } catch (err) {
        console.log(`Please configure rminfo.json to enable remote machine test.${err}`);
        skip = true;
    }

    before(async () => {
        chai.should();
        chai.use(chaiAsPromised);
        await cpp.exec(`echo '1234' > ${LOCALFILE}`);
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
        fs.unlinkSync(LOCALFILE);
    });

    it('Test mkdir', (done) => {
        if (skip) {
            done();

            return;
        }
        const conn: Client = new Client();
        conn.on('ready', async () => {
            const shellExecutor: ShellExecutor = new ShellExecutor(conn);
            await shellExecutor.initialize();
            let result = await shellExecutor.createFolder(REMOTEFOLDER, false);
            chai.expect(result).eq(true);
            result = await shellExecutor.removeFolder(REMOTEFOLDER);
            chai.expect(result).eq(true);
            done();
        }).connect(rmMeta);
    });

    it('Test ShellExecutor', (done) => {
        if (skip) {
            done();

            return;
        }
        const conn: Client = new Client();
        conn.on('ready', async () => {
            const shellExecutor: ShellExecutor = new ShellExecutor(conn);
            await shellExecutor.initialize();
            await copyFile(shellExecutor);
            await Promise.all([
                copyFileToRemoteLoop(shellExecutor),
                copyFileToRemoteLoop(shellExecutor),
                copyFileToRemoteLoop(shellExecutor),
                getRemoteFileContentLoop(shellExecutor)
            ]);
            done();
        }).connect(rmMeta);
    });
});
