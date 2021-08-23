// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { Channel, EnvironmentInformation, EnvironmentService, EnvironmentStatus } from "../environment";
import { EventEmitter } from 'events';
import { UtCommandChannel } from "./utCommandChannel";

export class UtEnvironmentService extends EnvironmentService {
    private allEnvironments = new Map<string, EnvironmentInformation>();
    private hasMoreEnvironmentsInternal = true;

    constructor() {
        super();
    }

    public get hasStorageService(): boolean {
        // storage service is tested by integration testing.
        return false;
    }

    public get useSharedStorage(): boolean {
        return false;
    }

    public get environmentMaintenceLoopInterval(): number {
        return 1;
    }

    public get getName(): string {
        return 'ut';
    }

    public initCommandChannel(eventEmitter: EventEmitter): void {
        this.commandChannel = new UtCommandChannel(eventEmitter);
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

    public testSetNoMoreEnvironment(hasMore: boolean): void {
        this.hasMoreEnvironmentsInternal = hasMore;
    }

    public get hasMoreEnvironments(): boolean {
        return this.hasMoreEnvironmentsInternal;
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
