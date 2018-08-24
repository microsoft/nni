/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import { Client } from 'ssh2';
import { Deferred } from 'ts-deferred';
import { SSHClientUtility } from '../remote_machine/sshClientUtility';

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
    let skip: boolean = true;
    let rmMeta: any;
    try {
        rmMeta = JSON.parse(fs.readFileSync('../../.vscode/rminfo.json', 'utf8'));
    } catch (err) {
        skip = true;
    }

    before(async () => {
        await cpp.exec(`echo '1234' > ${LOCALFILE}`);
    });

    after(() => {
        fs.unlinkSync(LOCALFILE);
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
