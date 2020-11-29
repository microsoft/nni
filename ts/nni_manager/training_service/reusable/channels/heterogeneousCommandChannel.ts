// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from "events";
import { delay } from "../../../common/utils";
import { CommandChannel, RunnerConnection } from "../commandChannel";
import { Channel, EnvironmentInformation } from "../environment";
import { AMLCommandChannel } from "./amlCommandChannel";
import { WebCommandChannel, WebRunnerConnection } from "./webCommandChannel";


export class HeterogenousCommandChannel extends CommandChannel{
    private stopping: boolean = false;
    private amlCommandChannel: AMLCommandChannel | undefined;
    private webCommandChannel: WebCommandChannel | undefined;

    public get channelName(): Channel {
        return "web";
    }

    public constructor(commandEmitter: EventEmitter, platformsArray: string[]) {
        super(commandEmitter);
        console.log(platformsArray.includes('local'))
        if (platformsArray.includes('local') || 
            platformsArray.includes('remote') || 
            platformsArray.includes('pai')) {
            this.webCommandChannel = new WebCommandChannel(commandEmitter);
        }
        if (platformsArray.includes('aml')) {
            this.amlCommandChannel = new AMLCommandChannel(commandEmitter);
        }
    }

    public async config(_key: string, _value: any): Promise<void> {
        // do nothing
    }

    public async start(): Promise<void> {
        const tasks: Promise<void>[] = [];
        if (this.amlCommandChannel) {
            tasks.push(this.amlCommandChannel.start());
        }
        if (this.webCommandChannel) {
            tasks.push(this.webCommandChannel.start());
        }
        await Promise.all(tasks);
    }

    public async stop(): Promise<void> {
        this.stopping = true;
    }

    public async open(environment: EnvironmentInformation): Promise<void> {
        const tasks: Promise<void>[] = [];
        if (this.amlCommandChannel) {
            tasks.push(this.amlCommandChannel.open(environment));
        }
        if (this.webCommandChannel) {
            tasks.push(this.webCommandChannel.open(environment));
        }
        await Promise.all(tasks);
    }

    public async close(environment: EnvironmentInformation): Promise<void> {
        const tasks: Promise<void>[] = [];
        if (this.amlCommandChannel) {
            tasks.push(this.amlCommandChannel.close(environment));
        }
        if (this.webCommandChannel) {
            tasks.push(this.webCommandChannel.close(environment));
        }
        await Promise.all(tasks);
    }

    public async run(): Promise<void> {
        const tasks: Promise<void>[] = [];
        if (this.amlCommandChannel) {
            tasks.push(this.amlCommandChannel.run());
        }
        if (this.webCommandChannel) {
            tasks.push(this.webCommandChannel.run());
        }
        await Promise.all(tasks);
    }

    protected async sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void> {
        switch (environment.platform) {
            case 'aml':
                if (this.amlCommandChannel === undefined) {
                    throw new Error(`amlCommandChannel not initialezed!`);
                }
                await this.amlCommandChannel.sendCommandInternal(environment, message);
                break;
            case 'local':
            case 'pai':
            case 'remote':
                if (this.webCommandChannel === undefined) {
                    throw new Error(`webCommandChannel not initialezed!`);
                }
                await this.webCommandChannel.sendCommandInternal(environment, message);
                break;
            default:
                throw new Error(`Heterogenous not support platform: '${environment.platform}'`);
        }
    }

    protected createRunnerConnection(environment: EnvironmentInformation): RunnerConnection {
        if (this.webCommandChannel) {
            return this.webCommandChannel.createRunnerConnection(environment);
        }
        return new WebRunnerConnection(environment);
    }
}
