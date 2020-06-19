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
import { EnvironmentInformation } from "./environment";
import { Logger, getLogger } from "../../common/log";

const acceptedCommands: Set<string> = new Set<string>(TRIAL_COMMANDS);

export enum ChannelType {
    API = "api",
    Storage = "storage",
}

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

export abstract class CommandChannel {
    protected readonly log: Logger;

    protected readonly commandEmitter: EventEmitter;
    private readonly commandPattern: RegExp = /(?<type>[\w]{2})(?<length>[\d]{14})(?<data>.*)\n?/gm;

    public constructor(commandEmitter: EventEmitter) {
        this.log = getLogger();
        this.commandEmitter = commandEmitter;
    }

    public abstract start(): void;
    public abstract stop(): void;

    public abstract open(environment: EnvironmentInformation): Promise<void>;
    public abstract close(environment: EnvironmentInformation): Promise<void>;

    protected abstract sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void>;

    public async sendCommand(environment: EnvironmentInformation, commantType: string, data: any): Promise<void> {
        const command = encodeCommand(commantType, JSON.stringify(data));
        await this.sendCommandInternal(environment, command.toString("utf8"));
        this.log.debug(`CommandChannel: env ${environment.id} sent command: ${command}`);
    }

    protected handleCommand(environment: EnvironmentInformation, content: string): void {
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
                const command = new Command(environment, commandType, finalData);
                this.commandEmitter.emit("command", command);
                this.log.debug(`CommandChannel: env ${environment.id} emit command: ${commandType}, ${data}`);
            }
            matches = this.commandPattern.exec(content);
        }
    }
}
