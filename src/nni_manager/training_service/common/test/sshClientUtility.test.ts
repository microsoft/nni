// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';

import { Client } from 'ssh2';
import { Deferred } from 'ts-deferred';
import { SSHClientUtility } from '../sshClientUtility';
import { ShellExecutor } from '../shellExecutor';
import { prepareUnitTest, cleanupUnitTest } from '../../../common/utils';

const LOCALFILE: string = '/tmp/sshclientUTData';
const REMOTEFILE: string = '/tmp/sshclientUTData';

async function copyFile(conn: Client): Promise<void> {
    const deferred: Deferred<void> = new Deferred<void>();

    conn.sftp((err, sftp) => {
        if (err) {
            deferred.reject(err);

            return;
        }
        sftp.fastPut(
            LOCALFILE,
            REMOTEFILE, (fastPutErr: Error) => {
                sftp.end();
                if (fastPutErr) {
                    deferred.reject(fastPutErr);
                } else {
                    deferred.resolve();
                }
            }
        );
    });

    return deferred.promise;
}

async function copyFileToRemoteLoop(conn: Client): Promise<void> {
    for (let i: number = 0; i < 500; i++) {
        console.log(i);
        await SSHClientUtility.copyFileToRemote(LOCALFILE, REMOTEFILE, conn);
    }
}

async function remoteExeCommandLoop(conn: Client): Promise<void> {
    for (let i: number = 0; i < 500; i++) {
        console.log(i);
        await SSHClientUtility.remoteExeCommand('ls', conn);
    }
}

async function getRemoteFileContentLoop(conn: Client): Promise<void> {
    for (let i: number = 0; i < 500; i++) {
        console.log(i);
        await SSHClientUtility.getRemoteFileContent(REMOTEFILE, conn);
    }
}

describe('sshClientUtility test', () => {
    let skip: boolean = false;
    let rmMeta: any;
    try {
        rmMeta = JSON.parse(fs.readFileSync('../../.vscode/rminfo.json', 'utf8'));
        console.log(rmMeta);
    } catch (err) {
        console.log(`Please configure rminfo.json to enable remote machine unit test.${err}`);
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
            let result = await shellExecutor.createFolder("ut_tmp_folder", false);
            chai.expect(result).eq(true);
            result = await shellExecutor.removeFolder("ut_tmp_folder");
            chai.expect(result).eq(true);
            done();
        }).connect(rmMeta);
    });

    it('Test SSHClientUtility', (done) => {
        if (skip) {
            done();

            return;
        }
        const conn: Client = new Client();
        conn.on('ready', async () => {
            await copyFile(conn);
            await Promise.all([
                copyFileToRemoteLoop(conn),
                copyFileToRemoteLoop(conn),
                copyFileToRemoteLoop(conn),
                remoteExeCommandLoop(conn),
                getRemoteFileContentLoop(conn)
            ]);
            done();
        }).connect(rmMeta);
    });
});
