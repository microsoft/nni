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
import * as path from 'path';
import { Client, ClientChannel, SFTPWrapper } from 'ssh2';
import * as stream from "stream";
import { Deferred } from 'ts-deferred';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getExperimentRootDir } from '../../common/utils';
import { RemoteCommandResult } from './remoteMachineData';

/**
 *
 * Utility for frequent operations towards SSH client
 *
 */
export namespace SSHClientUtility {
    /**
     * Copy files and directories in local directory recursively to remote directory
     * @param localDirectory local diretory
     * @param remoteDirectory remote directory
     * @param sshClient SSH client
     */
    export async function copyDirectoryToRemote(localDirectory : string, remoteDirectory : string, sshClient : Client) : Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        const localCompressedDir: string = path.join(getExperimentRootDir(), 'directory.tar.gz');
        const remoteCompressedDir: string = path.join(remoteDirectory, 'directory.tar.gz');

        // Compress files in local directory to experiment root directory
        await cpp.exec(`tar -czf ${localCompressedDir} -C ${localDirectory} .`);
        // Copy the compressed file to remoteDirectory and delete it
        await copyFileToRemote(localCompressedDir, remoteCompressedDir, sshClient);
        await cpp.exec(`rm ${localCompressedDir}`);
        // Decompress the remote compressed file in and delete it
        await remoteExeCommand(`tar -oxzf ${remoteCompressedDir} -C ${remoteDirectory}`, sshClient);
        await remoteExeCommand(`rm ${remoteCompressedDir}`, sshClient);
        deferred.resolve();

        return deferred.promise;
    }

    /**
     * Copy local file to remote path
     * @param localFilePath the path of local file
     * @param remoteFilePath the target path in remote machine
     * @param sshClient SSH Client
     */
    export function copyFileToRemote(localFilePath : string, remoteFilePath : string, sshClient : Client) : Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        sshClient.sftp((err : Error, sftp : SFTPWrapper) => {
            if (err) {
                deferred.reject();
            }
            sftp.fastPut(localFilePath, remoteFilePath, (fastPutErr : Error) => {
                sftp.end();
                if (fastPutErr) {
                    deferred.reject();
                } else {
                    deferred.resolve('success');
                }
            });
        });

        return deferred.promise;
    }

    /**
     * Execute command on remote machine
     * @param command the command to execute remotely
     * @param client SSH Client
     */
    export function remoteExeCommand(command : string, client : Client): Promise<RemoteCommandResult> {
        const deferred : Deferred<RemoteCommandResult> = new Deferred<RemoteCommandResult>();
        let stdout: string = '';
        let stderr: string = '';
        let exitCode : number;

        client.exec(command, (err : Error, channel : ClientChannel) => {
            if (err) {
                deferred.reject(err);
            }

            channel.on('data', function(data : any, dataStderr : any) {
                if (dataStderr) {
                    stderr += data.toString();
                }
                else {
                    stdout += data.toString();
                }
            }).on('exit', (code, signal) => {
                exitCode = code as number;
                deferred.resolve({
                    stdout : stdout,
                    stderr : stderr,
                    exitCode : exitCode
                });
            });
        });

        return deferred.promise;
    }

    export function getRemoteFileContent(filePath: string, sshClient: Client): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        sshClient.sftp((err: Error, sftp : SFTPWrapper) => {
            if (err) {
                deferred.reject(new Error(`SFTP error: ${err.message}`));
            }
            try {
                const sftpStream : stream.Readable = sftp.createReadStream(filePath);

                let dataBuffer: string = '';
                sftpStream.on('data', (data : Buffer | string) => {
                    dataBuffer += data;
                }).on('error', (streamErr: Error) => {
                    deferred.reject(new NNIError(NNIErrorNames.NOT_FOUND, streamErr.message));
                }).on('end', () => {
                    deferred.resolve(dataBuffer);
                });
            } catch (error) {
                deferred.reject(new Error(`SFTP error: ${error.message}`));
            }
        });

        return deferred.promise;
    }
}
