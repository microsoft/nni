// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as component from "../../../common/component";
import { delay } from "../../../common/utils";
import { CommandChannel, RunnerConnection } from "../commandChannel";
import { EnvironmentInformation, Channel } from "../environment";
import { AMLEnvironmentInformation } from '../aml/amlConfig';
import { EventEmitter } from 'events';
import { AMLEnvironmentService } from "../environments/amlEnvironmentService";
import { STDOUT } from "../../../core/commands";

class AMLRunnerConnection extends RunnerConnection {
}

export class AMLCommandChannel extends CommandChannel {
    private stopping: boolean = false;
    private currentMessageIndex: number = -1;
    private sendQueues: [EnvironmentInformation, string][] = [];
    private metricEmitter: EventEmitter | undefined;
    private readonly NNI_METRICS_PATTERN: string = `NNISDK_MEb'(?<metrics>.*?)'`;
    
    public constructor(commandEmitter: EventEmitter) {
        super(commandEmitter);
    }
    public get channelName(): Channel {
        return "aml";
    }

    public async config(_key: string, _value: any): Promise<void> {
        switch (_key) {
            case "MetricEmitter":
                this.metricEmitter = _value as EventEmitter;
                break;
        }
    }

    public async start(): Promise<void> {
        // start command loops
        await Promise.all([
            this.receiveLoop(),
            this.sendLoop()]);
    }

    public async stop(): Promise<void> {
        this.stopping = true;
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
                let command = await amlClient.receiveCommand();
                if (command && command.hasOwnProperty('trial_runner')) {
                    let messages = command['trial_runner'];
                    if (messages) {
                        if (messages instanceof Object && this.currentMessageIndex < messages.length - 1) {
                            for (let index = this.currentMessageIndex + 1; index < messages.length; index ++) {
                                this.handleTrialMessage(runnerConnection.environment, messages[index].toString());
                            }
                            this.currentMessageIndex = messages.length - 1;
                        } else if (this.currentMessageIndex === -1){
                            this.handleTrialMessage(runnerConnection.environment, messages.toString());
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

    private handleTrialMessage(environment: EnvironmentInformation, message: string): void {
        const commands = this.parseCommands(message);
        if (commands.length > 0) {
            const commandType = commands[0][0];
            if (commandType === STDOUT) {
                this.handleTrialMetrics(commands[0][1]);
            } else {
                this.handleCommand(environment, message);
            }
        }
    }

    private handleTrialMetrics(message: any): void {
        let trialId = message['trialId'];
        let msg = message['msg'];
        let tag = message['tag'];
        if (tag === 'trial') {
            const metricsContent: any = msg.match(this.NNI_METRICS_PATTERN);
            if (metricsContent && metricsContent.groups) {
                const key: string = 'metrics';
                const metric = metricsContent.groups[key];
                if (!this.metricEmitter) {
                    throw Error('metricEmitter not initialized');
                }
                this.metricEmitter.emit('metric', {
                    id: trialId,
                    data: metric
                });
            }
        }
    }
}
