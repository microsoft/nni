// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { OsCommands } from "../osCommands";
import { RemoteCommandResult } from "../remoteMachineData";

class WindowsCommands extends OsCommands {

    protected pathSpliter: string = '\\';
    protected multiplePathSpliter: RegExp = new RegExp(`\\${this.pathSpliter}{2,}`);

    public getScriptExt(): string {
        return "cmd";
    }
    public generateStartScript(workingDirectory: string, trialJobId: string, experimentId: string,
        trialSequenceId: string, isMultiPhase: boolean, jobIdFileName: string,
        command: string, nniManagerAddress: string, nniManagerPort: number,
        nniManagerVersion: string, logCollection: string, codeFile: string, cudaVisibleSetting: string): string {
        return `echo off
            set NNI_PLATFORM=remote
            set NNI_SYS_DIR=${workingDirectory}
            set NNI_OUTPUT_DIR=${workingDirectory}
            set NNI_TRIAL_JOB_ID=${trialJobId}
            set NNI_EXP_ID=${experimentId}
            set NNI_TRIAL_SEQ_ID=${trialSequenceId}
            set MULTI_PHASE=${isMultiPhase}
            ${cudaVisibleSetting !== "" ? "set " + cudaVisibleSetting : ""}
            cd %NNI_SYS_DIR%

            python -c "import nni" 2>nul
            if not %ERRORLEVEL% EQU 0 (
                echo installing NNI as exit code of "import nni" is %ERRORLEVEL%
                python -m pip install --user --upgrade nni
            )

            echo save process id
            powershell (Get-WmiObject Win32_Process -Filter ProcessId=$PID).ParentProcessId >${jobIdFileName}

            echo starting script
            python -m nni_trial_tool.trial_keeper --trial_command "${command}" --nnimanager_ip "${nniManagerAddress}" --nnimanager_port "${nniManagerPort}" --nni_manager_version "${nniManagerVersion}" --log_collection "${logCollection}" 1>%NNI_OUTPUT_DIR%/trialkeeper_stdout 2>%NNI_OUTPUT_DIR%/trialkeeper_stderr

            echo save exit code and time
            echo|set /p="%ERRORLEVEL% " > ${codeFile}
            powershell -command "(((New-TimeSpan -Start (Get-Date "01/01/1970") -End (Get-Date).ToUniversalTime()).TotalMilliseconds).ToString("0"))" >> ${codeFile}`;
    }

    public generateGpuStatsScript(scriptFolder: string): string {
        return `powershell "$env:METRIC_OUTPUT_DIR=${scriptFolder};$app = Start-Process python -ArgumentList \`"-m nni_gpu_tool.gpu_metrics_collector\`" -passthru -NoNewWindow;Write $app.ID | Out-File ${scriptFolder}\\pid -NoNewline -encoding utf8"`;
    }

    public getTempPath(): string {
        return "echo %TEMP%";
    }

    public createFolder(folderName: string, sharedFolder: boolean = false): string {
        let command;
        if (sharedFolder) {
            command = `mkdir "${folderName}"\r\nICACLS "${folderName}" /grant "Users":F`;
        } else {
            command = `mkdir "${folderName}"`;
        }
        return command;
    }

    public allowPermission(isRecursive: boolean = false, ...folders: string[]): string {
        let commands: string = "";

        folders.forEach(folder => {
            commands += `ICACLS "${folder}" /grant "Users":F ${isRecursive ? "/T" : ""}\r\n`
        });
        return commands;
    }

    public removeFolder(folderName: string, isRecursive: boolean = false, isForce: boolean = true): string {
        let flags = '';
        if (isForce || isRecursive) {
            flags = `-${isRecursive ? 's' : ''}${isForce ? 'q' : ''} `;
        }

        const command = `rmdir ${flags}"${folderName}"`;
        return command;
    }

    public removeFiles(folderName: string, filePattern: string): string {
        const files = this.joinPath(folderName, filePattern);
        const command = `del "${files}"`;
        return command;
    }

    public readLastLines(fileName: string, lineCount: number = 1): string {
        const command = `powershell.exe Get-Content "${fileName}" -Tail ${lineCount}`;
        return command;
    }

    public isProcessAliveCommand(pidFileName: string): string {
        const command = `powershell.exe Get-Process -Id (get-content "${pidFileName}") -ErrorAction SilentlyContinue`;
        return command;
    }

    public isProcessAliveProcessOutput(commandResult: RemoteCommandResult): boolean {
        let result = true;
        if (commandResult.exitCode !== 0) {
            result = false;
        }
        return result;
    }

    public killChildProcesses(pidFileName: string): string {
        const command = `powershell "$ppid=(type ${pidFileName}); function Kill-Tree {Param([int]$subppid);` +
            `Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $subppid } | ForEach-Object { Kill-Tree $_.ProcessId }; ` +
            `if ($subppid -ne $ppid){Stop-Process -Id $subppid}}` +
            `kill-tree $ppid"`;
        return command;
    }

    public extractFile(tarFileName: string, targetFolder: string): string {
        const command = `tar -xf "${tarFileName}" -C "${targetFolder}"`;
        return command;
    }

    public executeScript(script: string, isFile: boolean): string {
        let command: string;
        if (isFile) {
            command = `${script}`;
        } else {
            command = `cmd /c ${script}`;
        }
        return command;
    }
}

export { WindowsCommands };
