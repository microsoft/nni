// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Client } from "ssh2";
import { SSHClientUtility } from "./sshClientUtility";
import { Deferred } from "ts-deferred";
import { RemoteCommandResult } from "./remoteMachineData";
import { OsCommands } from "./osCommands";
import { LinuxCommands } from "./extends/linuxCommands";
// import { MethodNotImplementedError } from "common/errors";
import { promises } from "fs";

class ShellExecutor {

    readonly client: Client;
    private osCommands: OsCommands | undefined;

    protected pathSpliter: string = '/';
    protected multiplePathSpliter: RegExp = new RegExp(`\\${this.pathSpliter}{2,}`);

    constructor(sshClient: Client) {
        this.client = sshClient;
    }

    public async initialize(): Promise<void> {
        // check system version
        let result = await SSHClientUtility.remoteExeCommand("ver", this.client);
        if (result.exitCode == 0 && result.stdout.search("Windows") > -1) {
            // not implement Windows commands yet.
            throw new Error("not implement Windows commands yet.");
        } else {
            this.osCommands = new LinuxCommands();
        }
    }

    public async createFolder(folderName: string, sharedFolder: boolean = false): Promise<boolean> {
        const commandText = this.osCommands!.createFolder(folderName, sharedFolder);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode >= 0;
        return result;
    }

    public async removeFolder(folderName: string, isRecursive: boolean = false, isForce: boolean = true): Promise<boolean> {
        const commandText = this.osCommands!.removeFolder(folderName, isRecursive, isForce);
        const commandResult = await this.execute(commandText);
        const result = commandResult.exitCode >= 0;
        return result;
    }

    public async exists(folderOrFileName: string, isRecursive: boolean): Promise<boolean> {
        return true;
    }

    public async removeFiles(folderOrFileName: string, filePattern: string, isRecursive: boolean): Promise<boolean> {
        return true;
    }

    public async execute(command: string | undefined, processOutput: ((input: RemoteCommandResult) => RemoteCommandResult) | undefined = undefined, useShell: boolean = false): Promise<RemoteCommandResult> {
        const deferred: Deferred<RemoteCommandResult> = new Deferred<RemoteCommandResult>();
        let result: RemoteCommandResult = await SSHClientUtility.remoteExeCommand(command!, this.client, useShell);
        if (processOutput != undefined) {
            result = processOutput(result);
        }
        deferred.resolve(result);
        return deferred.promise;
    }
}

export { ShellExecutor };
