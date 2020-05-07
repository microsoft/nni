// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { RemoteCommandResult } from "./remoteMachineData";

abstract class OsCommands {

    protected pathSpliter: string = '/';
    protected multiplePathSpliter: RegExp = new RegExp(`${this.pathSpliter}{2,}`);

    public abstract getScriptExt(): string;
    public abstract generateStartScript(workingDirectory: string, trialJobId: string, experimentId: string,
        trialSequenceId: string, isMultiPhase: boolean, jobIdFileName: string,
        command: string, nniManagerAddress: string, nniManagerPort: number,
        nniManagerVersion: string, logCollection: string, codeFile: string,
        cudaVisibleSetting: string): string;
    public abstract getTempPath(): string;
    public abstract createFolder(folderName: string, sharedFolder: boolean): string;
    public abstract allowPermission(isRecursive: boolean, ...folders: string[]): string;
    public abstract removeFolder(folderName: string, isRecursive: boolean, isForce: boolean): string;
    public abstract removeFiles(folderOrFileName: string, filePattern: string): string;
    public abstract readLastLines(fileName: string, lineCount: number): string;
    public abstract isProcessAliveCommand(pidFileName: string): string;
    public abstract isProcessAliveProcessOutput(result: RemoteCommandResult): boolean;
    public abstract killChildProcesses(pidFileName: string): string;
    public abstract extractFile(tarFileName: string, targetFolder: string): string;
    public abstract executeScript(script: string, isFile: boolean): string;

    public joinPath(...paths: string[]): string {
        let dir: string = paths.filter((path: any) => path !== '').join(this.pathSpliter);
        if (dir === '') {
            dir = '.';
        } else {
            dir = dir.replace(this.multiplePathSpliter, this.pathSpliter);
        }
        return dir;
    }
}

export { OsCommands };
