// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Client, ClientChannel, ConnectConfig, SFTPWrapper } from 'ssh2';
import * as stream from 'stream';
import { Deferred } from "ts-deferred";
import { getLogger, Logger } from '../../common/log';
import { uniqueString, randomInt } from '../../common/utils';
import { execRemove, tarAdd } from '../common/util';
import { LinuxCommands } from "./extends/linuxCommands";
import { WindowsCommands } from './extends/windowsCommands';
import { OsCommands } from "./osCommands";
import { RemoteCommandResult, RemoteMachineMeta } from "./remoteMachineData";
import { NNIError, NNIErrorNames } from '../../common/errors';

class ShellExecutor {
    public name: string = "";

    private readonly lineBreaker = new RegExp(`[\r\n]+`);
    private readonly maxUsageCount = 5;

    private osCommands: OsCommands | undefined;
    private usedCount: number = 0; //count the connection number of every client
    private readonly sshClient: Client;
    private readonly log: Logger;
    private tempPath: string = "";
    private isWindows: boolean = false;
    private channelDefaultOutputs: string[] = [];
    private preCommand: string | undefined;

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
            tryKeyboard: true,
        };
        this.preCommand = rmMeta.preCommand;
        this.name = `${rmMeta.username}@${rmMeta.ip}:${rmMeta.port}`;
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

                // detect default output and trying to remove it under windows.
                // Anaconda has this kind of output.
                let defaultResult = await this.execute("");
                if (defaultResult.stdout !== "") {
                    deferred.reject(new Error(`The windows remote node shouldn't output welcome message, below content should be removed from the command window! \n` +
                        `${defaultResult.stdout}`));
                }
                defaultResult = await this.execute("powershell -command \"\"");
                if (defaultResult.stdout !== "") {
                    this.channelDefaultOutputs.push(defaultResult.stdout);
                }
                this.log.debug(`set channelDefaultOutput to "${this.channelDefaultOutputs}"`);

                // parse temp folder to expand possible environment variables.
                const commandResult = await this.execute("echo %TEMP%");
                this.tempPath = commandResult.stdout.replace(this.lineBreaker, "");
            } else {
                this.osCommands = new LinuxCommands();
                // it's not stable to get tmp path by Linux command, like "echo /tmp" or "ld -d /tmp".
                // Sometime it returns empty back, so hard code tmp path here.
                this.tempPath = "/tmp";
            }

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

    public addUsage(): boolean {
        let isAddedSuccess = false;
        if (this.usedCount < this.maxUsageCount) {
            this.usedCount++;
            isAddedSuccess = true;
        }
        return isAddedSuccess;
    }

    public releaseUsage(): boolean {
        let canBeReleased = false;
        if (this.usedCount > 0) {
            this.usedCount--;
        }
        if (this.usedCount == 0) {
            canBeReleased = true;
        }
        return canBeReleased;
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
        const exitCodeFile = this.joinPath(workingDirectory, '.nni', 'code');
        const codeDir = this.getRemoteCodePath(experimentId);

        return this.osCommands.generateStartScript(workingDirectory, trialJobId, experimentId,
            trialSequenceId, isMultiPhase, jobIdFileName, command,
            nniManagerAddress, nniManagerPort, nniManagerVersion,
            logCollection, exitCodeFile, codeDir, cudaVisibleSetting);
    }

    public generateGpuStatsScript(experimentId: string): string {
        if (this.osCommands === undefined) {
            throw new Error("osCommands must be initialized!");
        }
        return this.osCommands.generateGpuStatsScript(this.getRemoteScriptsPath(experimentId));
    }

    public getTempPath(): string {
        if (this.tempPath === "") {
            throw new Error("tempPath must be initialized!");
        }
        return this.tempPath;
    }

    public getRemoteScriptsPath(experimentId: string): string {
        return this.joinPath(this.getRemoteExperimentRootDir(experimentId), 'scripts');
    }

    public getRemoteCodePath(experimentId: string): string {
        return this.joinPath(this.getRemoteExperimentRootDir(experimentId), 'nni-code');
    }

    public getRemoteExperimentRootDir(experimentId: string): string {
        return this.joinPath(this.tempPath, 'nni-experiments', experimentId);
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
        const result = commandResult.exitCode == 0;
        return result;
    }

    public async allowPermission(isRecursive: boolean = false, ...folders: string[]): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.allowPermission(isRecursive, ...folders);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode == 0;
        return result;
    }

    public async removeFolder(folderName: string, isRecursive: boolean = false, isForce: boolean = true): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.removeFolder(folderName, isRecursive, isForce);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode == 0;
        return result;
    }

    public async removeFiles(folderOrFileName: string, filePattern: string = ""): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.removeFiles(folderOrFileName, filePattern);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode == 0;
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

    public async killChildProcesses(pidFileName: string, killSelf: boolean = false): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.killChildProcesses(pidFileName, killSelf);
        const commandResult = await this.execute(commandText);
        return commandResult.exitCode == 0;
    }

    public async fileExist(filePath: string): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.fileExistCommand(filePath);
        const commandResult = await this.execute(commandText);   
        return commandResult.stdout !== undefined && commandResult.stdout.trim() === 'True';
    }

    public async extractFile(tarFileName: string, targetFolder: string): Promise<boolean> {
        const commandText = this.osCommands && this.osCommands.extractFile(tarFileName, targetFolder);
        const commandResult = await this.execute(commandText);
        return commandResult.exitCode == 0;
    }

    public async executeScript(script: string, isFile: boolean = false, isInteractive: boolean = false): Promise<RemoteCommandResult> {
        const commandText = this.osCommands && this.osCommands.executeScript(script, isFile);
        const commandResult = await this.execute(commandText, undefined, isInteractive);
        return commandResult;
    }

    /**
     * Copy local file to remote path
     * @param localFilePath the path of local file
     * @param remoteFilePath the target path in remote machine
     */
    public async copyFileToRemote(localFilePath: string, remoteFilePath: string): Promise<boolean> {
        const commandIndex = randomInt(10000);
        this.log.debug(`copyFileToRemote(${commandIndex}): localFilePath: ${localFilePath}, remoteFilePath: ${remoteFilePath}`);

        const deferred: Deferred<boolean> = new Deferred<boolean>();
        this.sshClient.sftp((err: Error, sftp: SFTPWrapper) => {
            if (err !== undefined && err !== null) {
                this.log.error(`copyFileToRemote(${commandIndex}): ${err}`);
                deferred.reject(err);

                return;
            }
            assert(sftp !== undefined);
            sftp.fastPut(localFilePath, remoteFilePath, (fastPutErr: Error) => {
                sftp.end();
                if (fastPutErr !== undefined && fastPutErr !== null) {
                    this.log.error(`copyFileToRemote(${commandIndex}) fastPutErr: ${fastPutErr}, ${localFilePath}, ${remoteFilePath}`);
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
     */
    public async copyDirectoryToRemote(localDirectory: string, remoteDirectory: string): Promise<void> {
        const tmpSuffix: string = uniqueString(5);
        const localTarPath: string = path.join(os.tmpdir(), `nni_tmp_local_${tmpSuffix}.tar.gz`);
        if (!this.osCommands) {
            throw new Error("osCommands must be initialized!");
        }
        const remoteTarPath: string = this.osCommands.joinPath(this.tempPath, `nni_tmp_remote_${tmpSuffix}.tar.gz`);

        // Create remote directory
        await this.createFolder(remoteDirectory);
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
        const commandIndex = randomInt(10000);
        this.log.debug(`getRemoteFileContent(${commandIndex}): filePath: ${filePath}`);
        const deferred: Deferred<string> = new Deferred<string>();
        this.sshClient.sftp((err: Error, sftp: SFTPWrapper) => {
            if (err !== undefined && err !== null) {
                this.log.error(`getRemoteFileContent(${commandIndex}) sftp: ${err}`);
                deferred.reject(new Error(`SFTP error: ${err}`));

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
                this.log.error(`getRemoteFileContent(${commandIndex}): ${error.message}`);
                sftp.end();
                deferred.reject(new Error(`SFTP error: ${error.message}`));
            }
        });

        return deferred.promise;
    }

    private async execute(command: string | undefined, processOutput: ((input: RemoteCommandResult) => RemoteCommandResult) | undefined = undefined, useShell: boolean = false): Promise<RemoteCommandResult> {
        const deferred: Deferred<RemoteCommandResult> = new Deferred<RemoteCommandResult>();
        let stdout: string = '';
        let stderr: string = '';
        let exitCode: number;

        const commandIndex = randomInt(10000);
        if(this.osCommands !== undefined){
            command = this.osCommands.addPreCommand(this.preCommand, command);
        }
        this.log.debug(`remoteExeCommand(${commandIndex}): [${command}]`);

        // Windows always uses shell, and it needs to disable to get it works.
        useShell = useShell && !this.isWindows;

        const callback = (err: Error, channel: ClientChannel): void => {
            if (err !== undefined && err !== null) {
                this.log.error(`remoteExeCommand(${commandIndex}): ${err.message}`);
                deferred.reject(err);
                return;
            }

            channel.on('data', (data: any) => {
                stdout += data;
            });
            channel.on('exit', (code: any) => {
                exitCode = <number>code;

                // remove default output to get stdout correct.
                if (this.channelDefaultOutputs.length > 0) {
                    let modifiedStdout = stdout;
                    this.channelDefaultOutputs.forEach(defaultOutput => {
                        if (modifiedStdout.startsWith(defaultOutput)) {
                            if (modifiedStdout.length > defaultOutput.length) {
                                modifiedStdout = modifiedStdout.substr(defaultOutput.length);
                            } else if (modifiedStdout.length === defaultOutput.length) {
                                modifiedStdout = "";
                            }
                        }
                    });
                    stdout = modifiedStdout;
                }

                this.log.debug(`remoteExeCommand(${commandIndex}) exit(${exitCode})\nstdout: ${stdout}\nstderr: ${stderr}`);
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
            channel.stderr.on('data', function (data: any) {
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
