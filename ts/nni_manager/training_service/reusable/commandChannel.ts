// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from "events";
import { getLogger, Logger } from "../../common/log";
import { TRIAL_COMMANDS } from "../../core/commands";
import { encodeCommand } from "../../core/ipcInterface";
import { Channel, EnvironmentInformation } from "./environment";

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

export class RunnerConnection {
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
    public abstract start(): Promise<void>;
    public abstract stop(): Promise<void>;

    // Pull-based command channels need loop to check messages, the loop should be started with await here.
    public abstract run(): Promise<void>;

    protected abstract sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void>;
    protected abstract createRunnerConnection(environment: EnvironmentInformation): RunnerConnection;

    public async sendCommand(environment: EnvironmentInformation, commandType: string, data: any): Promise<void> {
        const command = encodeCommand(commandType, JSON.stringify(data));
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
                const data: any = matches.groups["data"];
                if (dataLength !== data.length) {
                    throw new Error(`dataLength ${dataLength} not equal to actual length ${data.length}: ${data}`);
                }
                try {
                    const finalData = JSON.parse(data);
                    // to handle encode('utf8') of Python
                    commands.push([commandType, finalData]);
                } catch (error) {
                    this.log.error(`CommandChannel: error on parseCommands ${error}, original: ${matches.groups["data"]}`);
                    throw error;
                }
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
            this.log.trace(`CommandChannel: env ${environment.id} emit command: ${commandType}, ${data}.`);
        }
    }
}
