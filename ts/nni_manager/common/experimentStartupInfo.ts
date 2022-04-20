// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import globals from 'common/globals';

export class ExperimentStartupInfo {
    public experimentId: string = globals.args.experimentId;
    public newExperiment: boolean = (globals.args.action === 'create');
    public basePort: number = globals.args.port;
    public logDir: string = globals.paths.experimentRoot;
    public logLevel: string = globals.args.logLevel;
    public readonly: boolean = (globals.args.action === 'view');
    public dispatcherPipe: string | null = globals.args.dispatcherPipe ?? null;
    public platform: string = globals.args.mode as string;
    public urlprefix: string = globals.args.urlPrefix;

    public static getInstance(): ExperimentStartupInfo {
        return new ExperimentStartupInfo();
    }
}

export function getExperimentStartupInfo(): ExperimentStartupInfo {
    return new ExperimentStartupInfo();
}

export function getExperimentId(): string {
    return globals.args.experimentId;
}

export function getBasePort(): number {
    return globals.args.port;
}

export function isNewExperiment(): boolean {
    return globals.args.action === 'create';
}

export function getPlatform(): string {
    return globals.args.mode as string;
}

export function isReadonly(): boolean {
    return globals.args.action === 'view';
}

export function getDispatcherPipe(): string | null {
    return globals.args.dispatcherPipe ?? null;
}
