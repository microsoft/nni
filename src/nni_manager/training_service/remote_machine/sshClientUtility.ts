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

import * as assert from 'assert';
import * as cpp from 'child-process-promise';
import * as os from 'os';
import * as path from 'path';
import { Client, ClientChannel, SFTPWrapper } from 'ssh2';
import * as stream from 'stream';
import { Deferred } from 'ts-deferred';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getLogger, Logger } from '../../common/log';
import { getRemoteTmpDir, uniqueString, unixPathJoin } from '../../common/utils';
import { execRemove, tarAdd } from '../common/util';
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
    export async function copyDirectoryToRemote(localDirectory : string, remoteDirectory : string, sshClient : Client, remoteOS: string)
     : Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        const tmpTarName: string = `${uniqueString(10)}.tar.gz`;
        const localTarPath: string = path.join(os.tmpdir(), tmpTarName);
        const remoteTarPath: string = unixPathJoin(getRemoteTmpDir(remoteOS), tmpTarName);

        // Compress files in local directory to experiment root directory
        await tarAdd(localTarPath, localDirectory);
        // Copy the compressed file to remoteDirectory and delete it
        await copyFileToRemote(localTarPath, remoteTarPath, sshClient);
        await execRemove(localTarPath);
        // Decompress the remote compressed file in and delete it
        await remoteExeCommand(`tar -oxzf ${remoteTarPath} -C ${remoteDirectory}`, sshClient);
        await remoteExeCommand(`rm ${remoteTarPath}`, sshClient);
        deferred.resolve();

        return deferred.promise;
    }

    /**
     * Copy local file to remote path
     * @param localFilePath the path of local file
     * @param remoteFilePath the target path in remote machine
     * @param sshClient SSH Client
     */
    export function copyFileToRemote(localFilePath : string, remoteFilePath : string, sshClient : Client) : Promise<boolean> {
        const log: Logger = getLogger();
        log.debug(`copyFileToRemote: localFilePath: ${localFilePath}, remoteFilePath: ${remoteFilePath}`);
        assert(sshClient !== undefined);
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        sshClient.sftp((err : Error, sftp : SFTPWrapper) => {
            if (err !== undefined && err !== null) {
                log.error(`copyFileToRemote: ${err.message}, ${localFilePath}, ${remoteFilePath}`);
                deferred.reject(err);

                return;
            }
            assert(sftp !== undefined);
            sftp.fastPut(localFilePath, remoteFilePath, (fastPutErr : Error) => {
                sftp.end();
                if (fastPutErr !== undefined && fastPutErr !== null) {
                    deferred.reject(fastPutErr);
                } else {
                    deferred.resolve(true);
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
    // tslint:disable:no-unsafe-any no-any
    export function remoteExeCommand(command : string, client : Client): Promise<RemoteCommandResult> {
        const log: Logger = getLogger();
        log.debug(`remoteExeCommand: command: [${command}]`);
        const deferred : Deferred<RemoteCommandResult> = new Deferred<RemoteCommandResult>();
        let stdout: string = '';
        let stderr: string = '';
        let exitCode : number;

        client.exec(command, (err : Error, channel : ClientChannel) => {
            if (err !== undefined && err !== null) {
                log.error(`remoteExeCommand: ${err.message}`);
                deferred.reject(err);

                return;
            }

            channel.on('data', (data : any, dataStderr : any) => {
                if (dataStderr !== undefined && dataStderr !== null) {
                    stderr += data.toString();
                } else {
                    stdout += data.toString();
                }
            })
              .on('exit', (code : any, signal : any) => {
                exitCode = <number>code;
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
            if (err !== undefined && err !== null) {
                getLogger()
                  .error(`getRemoteFileContent: ${err.message}`);
                deferred.reject(new Error(`SFTP error: ${err.message}`));

                return;
            }
            try {
                const sftpStream : stream.Readable = sftp.createReadStream(filePath);

                let dataBuffer: string = '';
                sftpStream.on('data', (data : Buffer | string) => {
                    dataBuffer += data;
                })
                  .on('error', (streamErr: Error) => {
                    sftp.end();
                    deferred.reject(new NNIError(NNIErrorNames.NOT_FOUND, streamErr.message));
                })
                  .on('end', () => {
                    // sftp connection need to be released manually once operation is done
                    sftp.end();
                    deferred.resolve(dataBuffer);
                });
            } catch (error) {
                getLogger()
                  .error(`getRemoteFileContent: ${error.message}`);
                sftp.end();
                deferred.reject(new Error(`SFTP error: ${error.message}`));
            }
        });

        return deferred.promise;
    }
    // tslint:enable:no-unsafe-any no-any
}
