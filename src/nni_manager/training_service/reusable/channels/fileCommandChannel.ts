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
import { CommandChannel } from "../commandChannel";
import { EnvironmentInformation } from "../environment";
import { StorageService } from "../storageService";

class FileHandler {
    public fileName: string;
    public offset: number = 0;

    constructor(fileName: string) {
        this.fileName = fileName;
    }
}


class EnvironmentHandler {
    public environment: EnvironmentInformation;
    public handlers: Map<string, FileHandler> = new Map<string, FileHandler>();

    constructor(environment: EnvironmentInformation) {
        this.environment = environment;
    }
}

export class FileCommandChannel extends CommandChannel {
    private readonly commandPath = "commands";
    private stopping: boolean = false;
    // each node have a receiver
    private receive_handlers: Map<string, EnvironmentHandler> = new Map<string, EnvironmentHandler>();
    // make sure no concurrent issue when sending commands.
    private send_queues: [EnvironmentInformation, string][] = [];

    public start(): void {
        // start command loops
        this.receiveLoop();
        this.sendLoop();
    }

    public stop(): void {
        this.stopping = true;
    }

    public async open(environment: EnvironmentInformation): Promise<void> {
        if (this.receive_handlers.has(environment.id)) {
            throw new Error(`FileCommandChannel: env ${environment.id} is opened already, shouldn't be opened again.`);
        }
        this.receive_handlers.set(environment.id, new EnvironmentHandler(environment));
    }

    public async close(environment: EnvironmentInformation): Promise<void> {
        if (this.receive_handlers.has(environment.id)) {
            this.receive_handlers.delete(environment.id);
        }
    }

    protected async sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void> {
        this.send_queues.push([environment, message]);
    }

    private async sendLoop(): Promise<void> {
        const intervalSeconds = 0.5;
        while (!this.stopping) {
            const start = new Date();

            if (this.send_queues.length > 0) {
                const storageService = component.get<StorageService>(StorageService);

                while (this.send_queues.length > 0) {
                    const item = this.send_queues.shift();
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

            const envs = [...this.receive_handlers.values()];
            for (const environmentHandler of envs) {
                const envCommandFolder = storageService.joinPath(environmentHandler.environment.workingFolder, this.commandPath);
                // open new command files
                if (environmentHandler.handlers.size < environmentHandler.environment.nodeCount) {
                    // to find all node commands file
                    const commandFileNames = await storageService.listDirectory(envCommandFolder);
                    const toAddedFileNames = [];
                    for (const commandFileName of commandFileNames) {
                        if (commandFileName.startsWith("runner_commands") && !environmentHandler.handlers.has(commandFileName)) {
                            toAddedFileNames.push(commandFileName);
                        }
                    }

                    for (const toAddedFileName of toAddedFileNames) {
                        const fullPath = storageService.joinPath(envCommandFolder, toAddedFileName);
                        const fileHandler: FileHandler = new FileHandler(fullPath);
                        environmentHandler.handlers.set(toAddedFileName, fileHandler);
                        this.log.debug(`FileCommandChannel: added fileHandler env ${environmentHandler.environment.id} ${toAddedFileName}`);
                    }
                }

                // to loop all commands
                for (const fileHandler of environmentHandler.handlers.values()) {
                    const newContent = await storageService.readFileContent(fileHandler.fileName, fileHandler.offset, undefined);
                    if (newContent.length > 0) {
                        const commands = newContent.split('\n');
                        for (const command of commands) {
                            this.handleCommand(environmentHandler.environment, command);
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
