// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { OsCommands } from "../osCommands";
import { RemoteCommandResult } from "../remoteMachineData";

class LinuxCommands extends OsCommands {

    public getScriptExt(): string {
        return "sh";
    }

    public generateStartScript(workingDirectory: string, trialJobId: string, experimentId: string,
        trialSequenceId: string, isMultiPhase: boolean, jobIdFileName: string,
        command: string, nniManagerAddress: string, nniManagerPort: number,
        nniManagerVersion: string, logCollection: string, exitCodeFile: string,
        codeDir: string, cudaVisibleSetting: string): string {

        return `#!/bin/bash
            export NNI_PLATFORM=remote NNI_SYS_DIR=${workingDirectory} NNI_OUTPUT_DIR=${workingDirectory} NNI_TRIAL_JOB_ID=${trialJobId} \
            NNI_EXP_ID=${experimentId} NNI_TRIAL_SEQ_ID=${trialSequenceId} NNI_CODE_DIR=${codeDir}
            export MULTI_PHASE=${isMultiPhase}
            mkdir -p $NNI_SYS_DIR/code
            cp -r $NNI_CODE_DIR/. $NNI_SYS_DIR/code
            sh $NNI_SYS_DIR/install_nni.sh
            cd $NNI_SYS_DIR/code
            python3 -m nni.tools.trial_tool.trial_keeper --trial_command '${cudaVisibleSetting} ${command}' --nnimanager_ip '${nniManagerAddress}' \
                --nnimanager_port '${nniManagerPort}' --nni_manager_version '${nniManagerVersion}' \
                --job_id_file ${jobIdFileName} \
                --log_collection '${logCollection}' 1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr
            echo $? \`date +%s%3N\` >${exitCodeFile}`;
    }

    public generateGpuStatsScript(scriptFolder: string): string {
        return `echo $$ > ${scriptFolder}/pid ; METRIC_OUTPUT_DIR=${scriptFolder} python3 -m nni.tools.gpu_tool.gpu_metrics_collector`;
    }

    public createFolder(folderName: string, sharedFolder: boolean = false): string {
        let command;
        if (sharedFolder) {
            command = `umask 0; mkdir -p '${folderName}'`;
        } else {
            command = `mkdir -p '${folderName}'`;
        }
        return command;
    }

    public allowPermission(isRecursive: boolean = false, ...folders: string[]): string {
        const folderString = folders.join("' '");
        let command;

        if (isRecursive) {
            command = `chmod 777 -R '${folderString}'`;
        } else {
            command = `chmod 777 '${folderString}'`;
        }
        return command;
    }

    public removeFolder(folderName: string, isRecursive: boolean = false, isForce: boolean = true): string {
        let flags = '';
        if (isForce || isRecursive) {
            flags = `-${isRecursive ? 'r' : 'd'}${isForce ? 'f' : ''} `;
        }

        const command = `rm ${flags}'${folderName}'`;
        return command;
    }

    public removeFiles(folderName: string, filePattern: string): string {
        const files = this.joinPath(folderName, filePattern);
        const command = `rm '${files}'`;
        return command;
    }

    public readLastLines(fileName: string, lineCount: number = 1): string {
        const command = `tail -n ${lineCount} '${fileName}'`;
        return command;
    }

    public isProcessAliveCommand(pidFileName: string): string {
        const command = `kill -0 \`cat '${pidFileName}'\``;
        return command;
    }

    public isProcessAliveProcessOutput(commandResult: RemoteCommandResult): boolean {
        let result = true;
        if (commandResult.exitCode !== 0) {
            result = false;
        }
        return result;
    }

    public killChildProcesses(pidFileName: string, killSelf: boolean): string {
        // prevent trialkeeper to be killed, so it can save exit code.
        let command = `list_descendants ()
                {
                local children=$(ps -o pid= --ppid "$1")

                for pid in $children
                do
                    list_descendants "$pid"
                done

                echo "$children"
                }
            kill $(list_descendants \`cat '${pidFileName}'\`)`
        if (killSelf) {
            command += `\nkill \`cat '${pidFileName}'\``
        }
        return command;
    }

    public extractFile(tarFileName: string, targetFolder: string): string {
        const command = `tar -oxzf '${tarFileName}' -C '${targetFolder}'`;
        return command;
    }

    public executeScript(script: string, isFile: boolean): string {
        let command: string;
        if (isFile) {
            command = `bash '${script}'`;
        } else {
            script = script.replace(/"/g, '\\"');
            command = `bash -c "${script}"`;
        }
        return command;
    }

    public addPreCommand(preCommand: string | undefined, command: string | undefined): string | undefined{
        if (command === undefined || command === '' || preCommand === undefined || preCommand === ''){
            return command;
        } else {
            return `${preCommand} && ${command}`;
        }
    }

    public fileExistCommand(filePath: string): string {
        return `test -e ${filePath} && echo True || echo False`;
    }
}

export { LinuxCommands };
