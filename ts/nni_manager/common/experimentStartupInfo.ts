// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import path from 'path';

import type { NniManagerArgs } from 'common/globals/arguments';

let singleton: ExperimentStartupInfo | null = null;

export class ExperimentStartupInfo {

    public experimentId: string;
    public newExperiment: boolean;
    public basePort: number;
    public logDir: string = '';
    public logLevel: string;
    public readonly: boolean;
    public dispatcherPipe: string | null;
    public platform: string;
    public urlprefix: string;

    constructor(args: NniManagerArgs) {
        this.experimentId = args.experimentId;
        this.newExperiment = (args.action === 'create');
        this.basePort = args.port;
        this.logDir = path.join(args.experimentsDirectory, args.experimentId);  // TODO: handle in globals
        this.logLevel = args.logLevel;
        this.readonly = (args.action === 'view');
        this.dispatcherPipe = args.dispatcherPipe ?? null;
        this.platform = args.mode as string;
        this.urlprefix = args.urlPrefix;
    }

    public static getInstance(): ExperimentStartupInfo {
        assert.notEqual(singleton, null);
        return singleton!;
    }
}

export function getExperimentStartupInfo(): ExperimentStartupInfo {
    return ExperimentStartupInfo.getInstance();
}

export function setExperimentStartupInfo(args: NniManagerArgs): void {
    singleton = new ExperimentStartupInfo(args);
}

export function getExperimentId(): string {
    return getExperimentStartupInfo().experimentId;
}

export function getBasePort(): number {
    return getExperimentStartupInfo().basePort;
}

export function isNewExperiment(): boolean {
    return getExperimentStartupInfo().newExperiment;
}

export function getPlatform(): string {
    return getExperimentStartupInfo().platform;
}

export function isReadonly(): boolean {
    return getExperimentStartupInfo().readonly;
}

export function getDispatcherPipe(): string | null {
    return getExperimentStartupInfo().dispatcherPipe;
}
