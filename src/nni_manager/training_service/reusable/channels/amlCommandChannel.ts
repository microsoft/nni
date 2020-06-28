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
import { EventEmitter } from 'events';

class AMLRunnerConnection extends RunnerConnection {
    
}

export class AMLCommandChannel extends CommandChannel {
    private stopping: boolean = false;
    private currentMessageIndex: number = -1;
    private currentMetricIndex: number = -1;
    // make sure no concurrent issue when sending commands.
    private sendQueues: [EnvironmentInformation, string][] = [];
    private readonly metricEmitter: EventEmitter;
    private readonly NNI_METRICS_PATTERN: string = `NNISDK_MEb'(?<metrics>.*?)'`;
    
    public constructor(commandEmitter: EventEmitter, metricsEmitter: EventEmitter) {
        super(commandEmitter);
        this.metricEmitter = metricsEmitter;
    }
    public get channelName(): Channel {
        return "aml";
    }

    public async config(_key: string, _value: any): Promise<void> {
        // do nothing
    }

    public async start(): Promise<void> {
        // start command loops
        this.receiveLoop();
        this.sendLoop();
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
            let sendCount = 0;
            if (this.sendQueues.length > 0) {
                while (this.sendQueues.length > 0) {
                    const item = this.sendQueues.shift();
                    if (item === undefined) {
                        break;
                    }
                    const environment = item[0];
                    const message = item[1];
                    const amlClient = environment.environmentClient;
                    amlClient.sendCommand(message);
                    // send command
                    sendCount += 1;
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
                const amlClient = runnerConnection.environment.environmentClient;
                let command = await amlClient.receiveCommand();
                if (command && command.hasOwnProperty('trial_runner')) {
                    let messages = command['trial_runner'];
                    if (messages) {
                        if (messages instanceof Object && this.currentMessageIndex < messages.length - 1) {
                            for (let index = this.currentMessageIndex + 1; index < messages.length; index ++) {
                                this.handleCommand(runnerConnection.environment, messages[index].toString());
                            }
                            this.currentMessageIndex = messages.length - 1;
                        } else if (this.currentMessageIndex === -1){
                            this.handleCommand(runnerConnection.environment, messages.toString());
                            this.currentMessageIndex += 1;
                        }
                    }
                } 
                if (command && command.hasOwnProperty('trial_runner_sdk')) {
                    let messages = command['trial_runner_sdk'];
                    if (messages) {
                        if (messages instanceof Object && this.currentMetricIndex < messages.length - 1) {
                            for (let index = this.currentMetricIndex + 1; index < messages.length; index ++) {
                                this.handleTrialMetrics(messages[index].toString());
                            }
                            this.currentMetricIndex = messages.length - 1;
                        } else if (this.currentMetricIndex === -1){
                            this.handleTrialMetrics(messages.toString());
                            this.currentMetricIndex += 1;
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

    private handleTrialMetrics(message: string): void {
        console.log('-------handle trial metric------' + message)
        let messageObj = JSON.parse(message);
        let trialId = messageObj['trialId'];
        let msg = messageObj['msg'];
        const metricsContent: any = msg.match(this.NNI_METRICS_PATTERN);
        if (metricsContent && metricsContent.groups) {
            const key: string = 'metrics';
            const metric = metricsContent.groups[key];
            console.log(`-----get ${metric} for trial ${trialId}------`);
            this.metricEmitter.emit('metric', {
                id: trialId,
                data: metric
            });
        }
    }
}
