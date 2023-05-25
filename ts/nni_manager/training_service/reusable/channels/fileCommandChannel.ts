// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { IocShim } from 'common/ioc_shim';
import { delay } from "common/utils";
import { CommandChannel, RunnerConnection } from "../commandChannel";
import { Channel, EnvironmentInformation } from "../environment";
import { StorageService } from "../storageService";

class FileHandler {
    public fileName: string;
    public offset: number = 0;

    constructor(fileName: string) {
        this.fileName = fileName;
    }
}


class FileRunnerConnection extends RunnerConnection {
    public handlers: Map<string, FileHandler> = new Map<string, FileHandler>();
}

export class FileCommandChannel extends CommandChannel {
    private readonly commandPath = "commands";
    private stopping: boolean = false;
    // make sure no concurrent issue when sending commands.
    private sendQueues: [EnvironmentInformation, string][] = [];

    public get channelName(): Channel {
        return "file";
    }

    public async config(_key: string, _value: any): Promise<void> {
        // do nothing
    }

    public async start(): Promise<void> {
        // do nothing
    }

    public async stop(): Promise<void> {
        this.stopping = true;
    }

    public async run(): Promise<void> {
        // start command loops
        await Promise.all([
            this.receiveLoop(),
            this.sendLoop()
        ]);
    }

    protected async sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void> {
        this.sendQueues.push([environment, message]);
    }

    protected createRunnerConnection(environment: EnvironmentInformation): RunnerConnection {
        return new FileRunnerConnection(environment);
    }

    private async sendLoop(): Promise<void> {
        const intervalSeconds = 0.5;
        while (!this.stopping) {
            const start = new Date();

            if (this.sendQueues.length > 0) {
                const storageService = IocShim.get<StorageService>(StorageService);

                while (this.sendQueues.length > 0) {
                    const item = this.sendQueues.shift();
                    if (item === undefined) {
                        break;
                    }
                    const environment = item[0];
                    const message = `${item[1]}\n`;

                    const fileName = storageService.joinPath(environment.workingFolder, this.commandPath, `manager_commands.txt`);
                    await storageService.save(message, fileName, true);
                }
            }

            const end = new Date();
            const delayMs = intervalSeconds * 1000 - (end.valueOf() - start.valueOf());
            if (delayMs > 0) {
                await delay(delayMs);
            }
        }
    }

    private async receiveLoop(): Promise<void> {
        const intervalSeconds = 2;
        const storageService = IocShim.get<StorageService>(StorageService);

        while (!this.stopping) {
            const start = new Date();

            const runnerConnections = [...this.runnerConnections.values()] as FileRunnerConnection[];
            for (const runnerConnection of runnerConnections) {
                const envCommandFolder = storageService.joinPath(runnerConnection.environment.workingFolder, this.commandPath);
                // open new command files
                if (runnerConnection.handlers.size < runnerConnection.environment.nodeCount) {
                    // to find all node commands file
                    const commandFileNames = await storageService.listDirectory(envCommandFolder);
                    const toAddedFileNames = [];
                    for (const commandFileName of commandFileNames) {
                        if (commandFileName.startsWith("runner_commands") && !runnerConnection.handlers.has(commandFileName)) {
                            toAddedFileNames.push(commandFileName);
                        }
                    }

                    for (const toAddedFileName of toAddedFileNames) {
                        const fullPath = storageService.joinPath(envCommandFolder, toAddedFileName);
                        const fileHandler: FileHandler = new FileHandler(fullPath);
                        runnerConnection.handlers.set(toAddedFileName, fileHandler);
                        this.log.debug(`FileCommandChannel: added fileHandler env ${runnerConnection.environment.id} ${toAddedFileName}`);
                    }
                }

                // to loop all commands
                for (const fileHandler of runnerConnection.handlers.values()) {
                    const newContent = await storageService.readFileContent(fileHandler.fileName, fileHandler.offset, undefined);
                    if (newContent.length > 0) {
                        const commands = newContent.split('\n');
                        for (const command of commands) {
                            this.handleCommand(runnerConnection.environment, command);
                        }
                        fileHandler.offset += newContent.length;
                    }
                }
            }

            const end = new Date();
            const delayMs = intervalSeconds * 1000 - (end.valueOf() - start.valueOf());
            if (delayMs > 0) {
                await delay(delayMs);
            }
        }
    }
}
