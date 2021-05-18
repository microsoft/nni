// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import * as os from 'os';
import * as path from 'path';
import * as component from '../common/component';

export class ExperimentStartupInfo {
    public experimentId: string;
    public newExperiment: boolean;
    public basePort: number;
    public logDir: string;
    public logLevel: string = '';
    public readonly: boolean = false;
    public dispatcherPipe: string | null = null;
    public platform: string;

    constructor(
            newExperiment: boolean,
            experimentId: string,
            basePort: number,
            platform: string,
            logDir?: string,
            logLevel?: string,
            readonly?: boolean,
            dispatcherPipe?: string) {

        assert(experimentId.trim().length > 0);
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
    }
}

export function setExperimentStartupInfo(
        newExperiment: boolean,
        experimentId: string,
        basePort: number,
        platform: string,
        logDir?: string,
        logLevel?: string,
        readonly?: boolean,
        dispatcherPipe?: string): void {

    (global as any).experimentStartupInfo = new ExperimentStartupInfo(
        newExperiment,
        experimentId,
        basePort,
        platform,
        logDir,
        logLevel,
        !!readonly,
        dispatcherPipe
    );
}

export function getExperimentStartupInfo(): ExperimentStartupInfo {
    assert((global as any).experimentStartupInfo !== null);
    return (global as any).experimentStartupInfo;
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
