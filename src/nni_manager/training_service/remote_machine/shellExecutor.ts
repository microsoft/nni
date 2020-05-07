// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { Client, ClientChannel, SFTPWrapper, ConnectConfig } from 'ssh2';
import { Deferred } from "ts-deferred";
import { RemoteCommandResult, RemoteMachineMeta } from "./remoteMachineData";
import * as stream from 'stream';
import { OsCommands } from "./osCommands";
import { LinuxCommands } from "./extends/linuxCommands";
import { WindowsCommands } from './extends/windowsCommands';
import { getLogger, Logger } from '../../common/log';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { execRemove, tarAdd } from '../common/util';
import { uniqueString } from '../../common/utils';

class ShellExecutor {
    private readonly lineBreaker = new RegExp(`[\r\n]+`);

    private osCommands: OsCommands | undefined;
    private usedConnectionNumber: number = 0; //count the connection number of every client
    private readonly sshClient: Client;
    private readonly log: Logger;
    private username: string = "";
    private tempPath: string = "";
    private isWindows: boolean = false;

    constructor() {
        this.log = getLogger();
        this.sshClient = new Client();
    }

    public async initialize(rmMeta: RemoteMachineMeta): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

        const connectConfig: ConnectConfig = {
            host: rmMeta.ip,
            port: rmMeta.port,
            username: rmMeta.username,
            tryKeyboard: true
        };
        this.username = rmMeta.username;
        if (rmMeta.passwd !== undefined) {
            connectConfig.password = rmMeta.passwd;
        } else if (rmMeta.sshKeyPath !== undefined) {
            if (!fs.existsSync(rmMeta.sshKeyPath)) {
                //SSh key path is not a valid file, reject
                deferred.reject(new Error(`${rmMeta.sshKeyPath} does not exist.`));
            }
            const privateKey: string = fs.readFileSync(rmMeta.sshKeyPath, 'utf8');

            connectConfig.privateKey = privateKey;
            connectConfig.passphrase = rmMeta.passphrase;
        } else {
            deferred.reject(new Error(`No valid passwd or sshKeyPath is configed.`));
        }
        this.sshClient.on('ready', async () => {
            // check OS type: windows or else
            const result = await this.execute("ver");
            if (result.exitCode == 0 && result.stdout.search("Windows") > -1) {
                this.osCommands = new WindowsCommands();
                this.isWindows = true;
            } else {
                this.osCommands = new LinuxCommands();
            }
            // parse temp folder to expand possible environment variables.
            const commandText = this.osCommands && this.osCommands.getTempPath();
            const commandResult = await this.execute(commandText);
            this.tempPath = commandResult.stdout.replace(this.lineBreaker, "");

            deferred.resolve();
        }).on('error', (err: Error) => {
            // SSH connection error, reject with error message
            deferred.reject(new Error(err.message));
        }).on("keyboard-interactive", (_name, _instructions, _lang, _prompts, finish) => {
            finish([rmMeta.passwd]);
        }).connect(connectConfig);

        return deferred.promise;
    }

    public close(): void {
        this.sshClient.end();
    }

    public get getUsedConnectionNumber(): number {
        return this.usedConnectionNumber;
    }

    public addUsedConnectionNumber(): void {
        this.usedConnectionNumber += 1;
    }

    public minusUsedConnectionNumber(): void {
        this.usedConnectionNumber -= 1;
    }

    public getScriptName(mainName: string): string {
        if (this.osCommands === undefined) {
            throw new Error("osCommands must be initialized!");
        }
        return `${mainName}.${this.osCommands.getScriptExt()}`;
    }

    public generateStartScript(workingDirectory: string, trialJobId: string, experimentId: string,
        trialSequenceId: string, isMultiPhase: boolean,
        command: string, nniManagerAddress: string, nniManagerPort: number,
        nniManagerVersion: string, logCollection: string, cudaVisibleSetting: string): string {
        if (this.osCommands === undefined) {
            throw new Error("osCommands must be initialized!");
        }
        const jobIdFileName = this.joinPath(workingDirectory, '.nni', 'jobpid');
        const codeFile = this.joinPath(workingDirectory, '.nni', 'code');

        return this.osCommands.generateStartScript(workingDirectory, trialJobId, experimentId,
            trialSequenceId, isMultiPhase, jobIdFileName,
            command, nniManagerAddress, nniManagerPort,
            nniManagerVersion, logCollection, codeFile, cudaVisibleSetting);
    }

    public getTempPath(): string {
        if (this.tempPath === "") {
            throw new Error("tempPath must be initialized!");
        }
        return this.tempPath;
    }

    public getRemoteScriptsPath(): string {
        return this.joinPath(this.tempPath, this.username, 'nni', 'scripts');
    }

    public getRemoteExperimentRootDir(experimentId: string): string {
        return this.joinPath(this.tempPath, 'nni', 'experiments', experimentId);
    }

    public joinPath(...paths: string[]): string {
        if (!this.osCommands) {
            throw new Error("osCommands must be initialized!");
        }
        return this.osCommands.joinPath(...paths);
    }

    public async createFolder(folderName: string, sharedFolder: boolean = false): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.createFolder(folderName, sharedFolder);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode >= 0;
        return result;
    }

    public async allowPermission(isRecursive: boolean = false, ...folders: string[]): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.allowPermission(isRecursive, ...folders);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode >= 0;
        return result;
    }

    public async removeFolder(folderName: string, isRecursive: boolean = false, isForce: boolean = true): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.removeFolder(folderName, isRecursive, isForce);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode >= 0;
        return result;
    }

    public async removeFiles(folderOrFileName: string, filePattern: string = ""): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.removeFiles(folderOrFileName, filePattern);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode >= 0;
        return result;
    }

    public async readLastLines(fileName: string, lineCount: number = 1): Promise<string> {
        const commandText = this.osCommands && this.osCommands.readLastLines(fileName, lineCount);
        const commandResult = await this.execute(commandText);
        let result: string = "";
        if (commandResult !== undefined && commandResult.stdout !== undefined && commandResult.stdout.length > 0) {
            result = commandResult.stdout;
        }
        return result;
    }

    public async isProcessAlive(pidFileName: string): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.isProcessAliveCommand(pidFileName);
        const commandResult = await this.execute(commandText);
        const result = this.osCommands && this.osCommands.isProcessAliveProcessOutput(commandResult);
        return result !== undefined ? result : false;
    }

    public async killChildProcesses(pidFileName: string): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.killChildProcesses(pidFileName);
        const commandResult = await this.execute(commandText);
        return commandResult.exitCode == 0;
    }

    public async extractFile(tarFileName: string, targetFolder: string): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.extractFile(tarFileName, targetFolder);
        const commandResult = await this.execute(commandText);
        return commandResult.exitCode == 0;
    }

    public async executeScript(script: string, isFile: boolean, isInteractive: boolean = false): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.executeScript(script, isFile);
        const commandResult = await this.execute(commandText, undefined, isInteractive);
        return commandResult.exitCode == 0;
    }

    /**
     * Copy local file to remote path
     * @param localFilePath the path of local file
     * @param remoteFilePath the target path in remote machine
     */
    public async copyFileToRemote(localFilePath: string, remoteFilePath: string): Promise<boolean> {
        this.log.debug(`copyFileToRemote: localFilePath: ${localFilePath}, remoteFilePath: ${remoteFilePath}`);

        const deferred: Deferred<boolean> = new Deferred<boolean>();
        this.sshClient.sftp((err: Error, sftp: SFTPWrapper) => {
            if (err !== undefined && err !== null) {
                this.log.error(`copyFileToRemote: ${err.message}, ${localFilePath}, ${remoteFilePath}`);
                deferred.reject(err);

                return;
            }
            assert(sftp !== undefined);
            sftp.fastPut(localFilePath, remoteFilePath, (fastPutErr: Error) => {
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
     * Copy files and directories in local directory recursively to remote directory
     * @param localDirectory local diretory
     * @param remoteDirectory remote directory
     * @param sshClient SSH client
     */
    public async copyDirectoryToRemote(localDirectory: string, remoteDirectory: string): Promise<void> {
        const tmpSuffix: string = uniqueString(5);
        const localTarPath: string = path.join(os.tmpdir(), `nni_tmp_local_${tmpSuffix}.tar.gz`);
        if (!this.osCommands) {
            throw new Error("osCommands must be initialized!");
        }
        const remoteTarPath: string = this.osCommands.joinPath(this.tempPath, `nni_tmp_remote_${tmpSuffix}.tar.gz`);

        // Compress files in local directory to experiment root directory
        await tarAdd(localTarPath, localDirectory);
        // Copy the compressed file to remoteDirectory and delete it
        await this.copyFileToRemote(localTarPath, remoteTarPath);
        await execRemove(localTarPath);
        // Decompress the remote compressed file in and delete it
        await this.extractFile(remoteTarPath, remoteDirectory);
        await this.removeFiles(remoteTarPath);
    }

    public async getRemoteFileContent(filePath: string): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        this.sshClient.sftp((err: Error, sftp: SFTPWrapper) => {
            if (err !== undefined && err !== null) {
                getLogger()
                    .error(`getRemoteFileContent: ${err.message}`);
                deferred.reject(new Error(`SFTP error: ${err.message}`));

                return;
            }
            try {
                const sftpStream: stream.Readable = sftp.createReadStream(filePath);

                let dataBuffer: string = '';
                sftpStream.on('data', (data: Buffer | string) => {
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

    private async execute(command: string | undefined, processOutput: ((input: RemoteCommandResult) => RemoteCommandResult) | undefined = undefined, useShell: boolean = false): Promise<RemoteCommandResult> {
        this.log.debug(`remoteExeCommand: command: [${command}]`);
        const deferred: Deferred<RemoteCommandResult> = new Deferred<RemoteCommandResult>();
        let stdout: string = '';
        let stderr: string = '';
        let exitCode: number;

        // Windows always uses shell, and it needs to disable to get it works.
        useShell = useShell && !this.isWindows;
        const callback = (err: Error, channel: ClientChannel): void => {
            if (err !== undefined && err !== null) {
                this.log.error(`remoteExeCommand: ${err.message}`);
                deferred.reject(err);
                return;
            }

            channel.on('data', (data: any) => {
                stdout += data;
            });
            channel.on('exit', (code: any) => {
                exitCode = <number>code;
                this.log.debug(`remoteExeCommand exit(${exitCode})\nstdout: ${stdout}\nstderr: ${stderr}`);
                let result = {
                    stdout: stdout,
                    stderr: stderr,
                    exitCode: exitCode
                };

                if (processOutput != undefined) {
                    result = processOutput(result);
                }
                deferred.resolve(result);
            });
            channel.stderr.on('data', function (data) {
                stderr += data;
            });

            if (useShell) {
                channel.stdin.write(`${command}\n`);
                channel.end("exit\n");
            }

            return;
        };

        if (useShell) {
            this.sshClient.shell(callback);
        } else {
            this.sshClient.exec(command !== undefined ? command : "", callback);
        }

        return deferred.promise;
    }
}

export { ShellExecutor };
