// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import * as os from 'os';
import * as path from 'path';

const API_ROOT_URL: string = '/api/v1/nni';

let singleton: ExperimentStartupInfo | null = null;

export class ExperimentStartupInfo {

    public experimentId: string = '';
    public newExperiment: boolean = true;
    public basePort: number = -1;
    public initialized: boolean = false;
    public logDir: string = '';
    public logLevel: string = '';
    public readonly: boolean = false;
    public dispatcherPipe: string | null = null;
    public platform: string = '';
    public urlprefix: string = '';

    constructor(
            newExperiment: boolean,
            experimentId: string,
            basePort: number,
            platform: string,
            logDir?: string,
            logLevel?: string,
            readonly?: boolean,
            dispatcherPipe?: string,
            urlprefix?: string) {
        this.newExperiment = newExperiment;
        this.experimentId = experimentId;
        this.basePort = basePort;
        this.platform = platform;

        if (logDir !== undefined && logDir.length > 0) {
            this.logDir = path.join(path.normalize(logDir), experimentId);
        } else {
            this.logDir = path.join(os.homedir(), 'nni-experiments', experimentId);
        }

        if (logLevel !== undefined && logLevel.length > 1) {
            this.logLevel = logLevel;
        }

        if (readonly !== undefined) {
            this.readonly = readonly;
        }

        if (dispatcherPipe != undefined && dispatcherPipe.length > 0) {
            this.dispatcherPipe = dispatcherPipe;
        }

        if(urlprefix != undefined && urlprefix.length > 0){
            this.urlprefix = urlprefix;
        }
    }

    public get apiRootUrl(): string {
        return this.urlprefix === '' ? API_ROOT_URL : `/${this.urlprefix}${API_ROOT_URL}`;
    }

    public static getInstance(): ExperimentStartupInfo {
        assert(singleton !== null);
        return singleton!;
    }
}

export function getExperimentStartupInfo(): ExperimentStartupInfo {
    return ExperimentStartupInfo.getInstance();
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
    singleton = new ExperimentStartupInfo(
        newExperiment,
        experimentId,
        basePort,
        platform,
        logDir,
        logLevel,
        readonly,
        dispatcherPipe,
        urlprefix
    );
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

export function getAPIRootUrl(): string {
    return getExperimentStartupInfo().apiRootUrl;
}
