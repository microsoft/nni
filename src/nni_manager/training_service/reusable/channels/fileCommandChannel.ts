/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import * as component from "../../../common/component";
import { delay } from "../../../common/utils";
import { CommandChannel, RunnerConnection } from "../commandChannel";
import { EnvironmentInformation, Channel } from "../environment";
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

    public start(): void {
        // start command loops
        this.receiveLoop();
        this.sendLoop();
    }

    public stop(): void {
        this.stopping = true;
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
                const storageService = component.get<StorageService>(StorageService);

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
        const storageService = component.get<StorageService>(StorageService);

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
