// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { OsCommands } from "../osCommands";

class LinuxCommands extends OsCommands {
    protected pathSpliter: string = '/';

    public removeFiles(folderName: string, filePattern: string, isRecursive: boolean): string {
        throw new Error("Method not implemented.");
    }

    public createFolder(folderName: string, sharedFolder: boolean = false): string {
        let command;
        if (sharedFolder) {
            command = `umask 0; mkdir -p '${folderName}'`;
        } else {
            command = `mkdir -p '${folderName}'`;;
        }
        return command;
    }

    public removeFolder(folderName: string, isRecursive: boolean = false, isForce: boolean = true): string {
        let command;
        let flags = '';
        if (isForce || isRecursive) {
            flags = `-${isRecursive ? 'r' : 'd'}${isForce ? 'f' : ''} `;
        }

        command = `rm ${flags}'${folderName}'`;
        return command;
    }
}

export { LinuxCommands };
