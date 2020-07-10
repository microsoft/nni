// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { encodeCommand } from "../../../core/ipcInterface";
import { Command, CommandChannel, RunnerConnection } from "../commandChannel";
import { Channel, EnvironmentInformation } from "../environment";

class UtRunnerConnection extends RunnerConnection {

}

export class UtCommandChannel extends CommandChannel {
    private readonly receivedCommands: Command[] = [];

    public get channelName(): Channel {
        return "ut";
    }

    public async testSendCommandToTrialDispatcher(environment: EnvironmentInformation, commandType: string, commandData: any) {
        const content = encodeCommand(commandType, JSON.stringify(commandData));
        this.log.debug(`UtCommandChannel: env ${environment.id} send test command ${content}`);
        this.handleCommand(environment, content.toString("utf8"));
    }

    public async testReceiveCommandFromTrialDispatcher(): Promise<Command | undefined> {
        return this.receivedCommands.shift();
    }

    public async config(_key: string, value: any): Promise<void> {
        // do nothing
    }

    public async start(): Promise<void> {
        // do nothing
    }

    public async stop(): Promise<void> {
        // do nothing
    }

    public async run(): Promise<void> {
        // do nothing
    }

    protected async sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void> {
        const parsedCommands = this.parseCommands(message);
        for (const parsedCommand of parsedCommands) {
            const command = new Command(environment, parsedCommand[0], parsedCommand[1]);
            this.receivedCommands.push(command);
        }
    }

    protected createRunnerConnection(environment: EnvironmentInformation): RunnerConnection {
        // do nothing
        return new UtRunnerConnection(environment);
    }
}
