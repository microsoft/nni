// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from "events";
import { delay } from "../../../common/utils";
import { CommandChannel, RunnerConnection } from "../commandChannel";
import { Channel, EnvironmentInformation } from "../environment";
import { AMLCommandChannel } from "./amlCommandChannel";
import { WebCommandChannel } from "./webCommandChannel";

export class HeterogenousCommandChannel extends CommandChannel{
    private stopping: boolean = false;
    private amlCommandChannel: AMLCommandChannel | undefined;
    private webCommandChannel: WebCommandChannel | undefined;

    public get channelName(): Channel {
        return "heterogenous";
    }

    public constructor(commandEmitter: EventEmitter) {
        super(commandEmitter);
    }

    public async config(_key: string, _value: any): Promise<void> {
        // do nothing
    }

    public async start(): Promise<void> {
        if (this.amlCommandChannel) {
            this.amlCommandChannel.start();
        }
        if (this.webCommandChannel) {
            this.webCommandChannel.start();
        }
    }

    public async stop(): Promise<void> {
        this.stopping = true;
    }

    public async run(): Promise<void> {
        if (this.amlCommandChannel) {
            this.amlCommandChannel.run();
        }
        if (this.webCommandChannel) {
            this.webCommandChannel.run();
        }
    }

    protected async sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void> {
        switch (environment.platform) {
            case 'aml':
                if (this.amlCommandChannel === undefined) {
                    throw new Error(`amlCommandChannel not initialezed!`);
                }
                this.amlCommandChannel.sendCommandInternal(environment, message);
                break;
            case 'local':
            case 'pai':
            case 'remote':
                if (this.webCommandChannel === undefined) {
                    throw new Error(`webCommandChannel not initialezed!`);
                }
                this.webCommandChannel.sendCommandInternal(environment, message);
                break;
            default:
                throw new Error(`Heterogenous not support platform: '${environment.platform}'`);
        }
    }

    protected createRunnerConnection(environment: EnvironmentInformation): RunnerConnection {
        return new RunnerConnection(environment);
    }
}
