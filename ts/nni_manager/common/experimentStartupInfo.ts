// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert';
import os from 'os';
import path from 'path';

import globals from 'common/globals';

const API_ROOT_URL: string = '/api/v1/nni';

let singleton: ExperimentStartupInfo | null = null;

export class ExperimentStartupInfo {

    public experimentId: string;
    public newExperiment: boolean;
    public basePort: number;
    public logDir: string;
    public logLevel: string;
    public readonly: boolean;
    public dispatcherPipe: string | null;
    public platform: string;
    public urlprefix: string;

    constructor() {
        this.experimentId = globals.args.experimentId;
        this.newExperiment = globals.args.action === 'create';
        this.basePort = globals.args.port;
        this.logDir = globals.paths.experimentRoot;
        this.logLevel = globals.args.logLevel;
        this.readonly = globals.args.action === 'view';
        this.dispatcherPipe = globals.args.dispatcherPipe ?? null;
        this.platform = <string>globals.args.mode;
        this.urlprefix = globals.args.urlPrefix;
    }

    public static getInstance(): ExperimentStartupInfo {
        if (singleton === null) {
            singleton = new ExperimentStartupInfo();
        }
        return singleton;
    }
}

export function getExperimentStartupInfo(): ExperimentStartupInfo {
    return ExperimentStartupInfo.getInstance();
}

export function resetExperimentStartupInfo(): void {
    singleton = new ExperimentStartupInfo();
}

export function setExperimentStartupInfo(
        newExperiment: boolean,
        experimentId: string,
        basePort: number,
        platform: string,
        logDir?: string,
        logLevel?: string,
        readonly?: boolean,
        dispatcherPipe?: string,
        urlprefix?: string): void {
    singleton = <ExperimentStartupInfo>{
        newExperiment,
        experimentId,
        basePort,
        platform,
        logDir,
        logLevel,
        readonly,
        dispatcherPipe,
        urlprefix
    };
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
