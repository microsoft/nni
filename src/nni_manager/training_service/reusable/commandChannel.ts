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

import { EventEmitter } from "events";
import { TRIAL_COMMANDS } from "../../core/commands";
import { encodeCommand } from "../../core/ipcInterface";
import { EnvironmentInformation, Channel } from "./environment";
import { Logger, getLogger } from "../../common/log";

const acceptedCommands: Set<string> = new Set<string>(TRIAL_COMMANDS);

export class Command {
    public readonly environment: EnvironmentInformation;
    public readonly command: string;
    public readonly data: any;

    constructor(environment: EnvironmentInformation, command: string, data: any) {
        if (!acceptedCommands.has(command)) {
            throw new Error(`unaccepted command ${command}`);
        }
        this.environment = environment;
        this.command = command;
        this.data = data;
    }
}

export abstract class RunnerConnection {
    public readonly environment: EnvironmentInformation;

    constructor(environment: EnvironmentInformation) {
        this.environment = environment;
    }

    public async open(): Promise<void> {
        // do nothing
    }

    public async close(): Promise<void> {
        // do nothing
    }
}

export abstract class CommandChannel {
    protected readonly log: Logger;
    protected runnerConnections: Map<string, RunnerConnection> = new Map<string, RunnerConnection>();
    protected readonly commandEmitter: EventEmitter;

    private readonly commandPattern: RegExp = /(?<type>[\w]{2})(?<length>[\d]{14})(?<data>.*)\n?/gm;

    public constructor(commandEmitter: EventEmitter) {
        this.log = getLogger();
        this.commandEmitter = commandEmitter;
    }

    public abstract get channelName(): Channel;
    public abstract config(key: string, value: any): Promise<void>;
    public abstract start(): void;
    public abstract stop(): void;

    protected abstract sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void>;
    protected abstract createRunnerConnection(environment: EnvironmentInformation): RunnerConnection;

    public async sendCommand(environment: EnvironmentInformation, commantType: string, data: any): Promise<void> {
        const command = encodeCommand(commantType, JSON.stringify(data));
        this.log.debug(`CommandChannel: env ${environment.id} sending command: ${command}`);
        await this.sendCommandInternal(environment, command.toString("utf8"));
    }

    public async open(environment: EnvironmentInformation): Promise<void> {
        if (this.runnerConnections.has(environment.id)) {
            throw new Error(`CommandChannel: env ${environment.id} is opened already, shouldn't be opened again.`);
        }
        const connection = this.createRunnerConnection(environment);
        this.runnerConnections.set(environment.id, connection);
        await connection.open();
    }

    public async close(environment: EnvironmentInformation): Promise<void> {
        if (this.runnerConnections.has(environment.id)) {
            const connection = this.runnerConnections.get(environment.id);
            this.runnerConnections.delete(environment.id);
            if (connection !== undefined) {
                await connection.close();
            }
        }
    }

    protected parseCommands(content: string): [string, any][] {
        const commands: [string, any][] = [];

        let matches = this.commandPattern.exec(content);

        while (matches) {
            if (undefined !== matches.groups) {
                const commandType = matches.groups["type"];
                const dataLength = parseInt(matches.groups["length"]);
                let data: any = matches.groups["data"];
                if (dataLength !== data.length) {
                    throw new Error(`dataLength ${dataLength} not equal to actual length ${data.length}: ${data}`);
                }
                // to handle encode('utf8') of Python
                data = JSON.parse('"' + data.split('"').join('\\"') + '"');
                const finalData = JSON.parse(data);
                commands.push([commandType, finalData]);
            }
            matches = this.commandPattern.exec(content);
        }

        return commands;
    }

    protected handleCommand(environment: EnvironmentInformation, content: string): void {
        const parsedResults = this.parseCommands(content);

        for (const parsedResult of parsedResults) {
            const commandType = parsedResult[0];
            const data = parsedResult[1];
            const command = new Command(environment, commandType, data);
            this.commandEmitter.emit("command", command);
            this.log.trace(`CommandChannel: env ${environment.id} emit command: ${commandType}, ${data}`);
        }
    }
}
