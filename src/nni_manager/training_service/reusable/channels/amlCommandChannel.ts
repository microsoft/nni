// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { delay } from "../../../common/utils";
import { AMLEnvironmentInformation } from '../aml/amlConfig';
import { CommandChannel, RunnerConnection } from "../commandChannel";
import { Channel, EnvironmentInformation } from "../environment";

class AMLRunnerConnection extends RunnerConnection {
}

export class AMLCommandChannel extends CommandChannel {
    private stopping: boolean = false;
    private currentMessageIndex: number = -1;
    private sendQueues: [EnvironmentInformation, string][] = [];

    public get channelName(): Channel {
        return "aml";
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
        return new AMLRunnerConnection(environment);
    }

    private async sendLoop(): Promise<void> {
        const intervalSeconds = 0.5;
        while (!this.stopping) {
            const start = new Date();
            if (this.sendQueues.length > 0) {
                while (this.sendQueues.length > 0) {
                    const item = this.sendQueues.shift();
                    if (item === undefined) {
                        break;
                    }
                    const environment = item[0];
                    const message = item[1];
                    const amlClient = (environment as AMLEnvironmentInformation).amlClient;
                    if (!amlClient) {
                        throw new Error('aml client not initialized!');
                    }
                    amlClient.sendCommand(message);
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

        while (!this.stopping) {
            const start = new Date();
            const runnerConnections = [...this.runnerConnections.values()] as AMLRunnerConnection[];
            for (const runnerConnection of runnerConnections) {
                // to loop all commands
                const amlClient = (runnerConnection.environment as AMLEnvironmentInformation).amlClient;
                if (!amlClient) {
                    throw new Error('AML client not initialized!');
                }
                const command = await amlClient.receiveCommand();
                if (command && Object.prototype.hasOwnProperty.call(command, "trial_runner")) {
                    const messages = command['trial_runner'];
                    if (messages) {
                        if (messages instanceof Object && this.currentMessageIndex < messages.length - 1) {
                            for (let index = this.currentMessageIndex + 1; index < messages.length; index++) {
                                this.handleCommand(runnerConnection.environment, messages[index]);
                            }
                            this.currentMessageIndex = messages.length - 1;
                        } else if (this.currentMessageIndex === -1) {
                            this.handleCommand(runnerConnection.environment, messages);
                            this.currentMessageIndex += 1;
                        }
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
