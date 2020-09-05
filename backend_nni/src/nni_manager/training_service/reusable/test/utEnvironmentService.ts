// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EnvironmentInformation, EnvironmentService, EnvironmentStatus } from "../environment";
import { EventEmitter } from "events";
import { CommandChannel } from "../commandChannel";
import { UtCommandChannel } from "./utCommandChannel";

export class UtEnvironmentService extends EnvironmentService {
    private commandChannel: UtCommandChannel | undefined;
    private allEnvironments = new Map<string, EnvironmentInformation>();
    private hasMoreEnvironmentsInternal = true;

    constructor() {
        super();
    }

    public get hasStorageService(): boolean {
        // storage service is tested by integration testing.
        return false;
    }
    public get environmentMaintenceLoopInterval(): number {
        return 1;
    }

    public testSetEnvironmentStatus(environment: EnvironmentInformation, newStatus: EnvironmentStatus): void {
        environment.status = newStatus;
    }

    public testReset(): void {
        this.allEnvironments.clear();
    }

    public testGetEnvironments(): Map<string, EnvironmentInformation> {
        return this.allEnvironments;
    }

    public testGetCommandChannel(): UtCommandChannel {
        if (this.commandChannel === undefined) {
            throw new Error(`command channel shouldn't be undefined.`);
        }
        return this.commandChannel;
    }

    public testSetNoMoreEnvironment(hasMore: boolean): void {
        this.hasMoreEnvironmentsInternal = hasMore;
    }

    public get hasMoreEnvironments(): boolean {
        return this.hasMoreEnvironmentsInternal;
    }

    public createCommandChannel(commandEmitter: EventEmitter): CommandChannel {
        this.commandChannel = new UtCommandChannel(commandEmitter)
        return this.commandChannel;
    }

    public async config(_key: string, _value: string): Promise<void> {
        // do nothing
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        // do nothing
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (!this.allEnvironments.has(environment.id)) {
            this.allEnvironments.set(environment.id, environment);
            environment.status = "WAITING";
        }
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        environment.status = "USER_CANCELED";
    }
}
