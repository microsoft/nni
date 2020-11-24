// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { RemoteCommandResult } from "./remoteMachineData";

abstract class OsCommands {

    protected pathSpliter: string = '/';
    protected multiplePathSpliter: RegExp = new RegExp(`[\\\\/]{2,}`);
    protected normalizePath: RegExp = new RegExp(`[\\\\/]`);

    public abstract getScriptExt(): string;
    public abstract generateStartScript(workingDirectory: string, trialJobId: string, experimentId: string,
        trialSequenceId: string, isMultiPhase: boolean, jobIdFileName: string,
        command: string, nniManagerAddress: string, nniManagerPort: number,
        nniManagerVersion: string, logCollection: string, exitCodeFile: string,
        codeDir: string, cudaVisibleSetting: string): string;
    public abstract generateGpuStatsScript(scriptFolder: string): string;
    public abstract createFolder(folderName: string, sharedFolder: boolean): string;
    public abstract allowPermission(isRecursive: boolean, ...folders: string[]): string;
    public abstract removeFolder(folderName: string, isRecursive: boolean, isForce: boolean): string;
    public abstract removeFiles(folderOrFileName: string, filePattern: string): string;
    public abstract readLastLines(fileName: string, lineCount: number): string;
    public abstract isProcessAliveCommand(pidFileName: string): string;
    public abstract isProcessAliveProcessOutput(result: RemoteCommandResult): boolean;
    public abstract killChildProcesses(pidFileName: string, killSelf: boolean): string;
    public abstract extractFile(tarFileName: string, targetFolder: string): string;
    public abstract executeScript(script: string, isFile: boolean): string;
    public abstract addPreCommand(preCommand: string | undefined, command: string | undefined): string | undefined;
    public abstract fileExistCommand(filePath: string): string | undefined;

    public joinPath(...paths: string[]): string {
        let dir: string = paths.filter((path: any) => path !== '').join(this.pathSpliter);
        if (dir === '') {
            dir = '.';
        } else {
            // normalize
            dir = dir.replace(this.normalizePath, this.pathSpliter);
            // reduce duplicate ones
            dir = dir.replace(this.multiplePathSpliter, this.pathSpliter);
        }
        return dir;
    }
}

export { OsCommands };
