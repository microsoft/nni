// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { OsCommands } from "../osCommands";
import { RemoteCommandResult } from "../remoteMachineData";

class LinuxCommands extends OsCommands {
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
        if (commandResult.exitCode > 0) {
            result = false;
        }
        return result;
    }

    public killChildProcesses(pidFileName: string): string {
        const command = `pkill -P \`cat '${pidFileName}'\``;
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
            script = script.replace('"', '\\"');
            command = `bash -c "${script}"`;
        }
        return command;
    }
}

export { LinuxCommands };
