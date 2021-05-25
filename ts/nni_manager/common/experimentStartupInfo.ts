// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import * as os from 'os';
import * as path from 'path';
import * as component from '../common/component';

@component.Singleton
class ExperimentStartupInfo {
    private readonly API_ROOT_URL: string = '/api/v1/nni';

    private experimentId: string = '';
    private newExperiment: boolean = true;
    private basePort: number = -1;
    private initialized: boolean = false;
    private logDir: string = '';
    private logLevel: string = '';
    private readonly: boolean = false;
    private dispatcherPipe: string | null = null;
    private platform: string = '';
    private urlprefix: string = '';

    public setStartupInfo(newExperiment: boolean, experimentId: string, basePort: number, platform: string, logDir?: string, logLevel?: string, readonly?: boolean, dispatcherPipe?: string, urlprefix?: string): void {
        assert(!this.initialized);
        assert(experimentId.trim().length > 0);
        this.newExperiment = newExperiment;
        this.experimentId = experimentId;
        this.basePort = basePort;
        this.initialized = true;
        this.platform = platform;

        if (logDir !== undefined && logDir.length > 0) {
            this.logDir = path.join(path.normalize(logDir), this.getExperimentId());
        } else {
            this.logDir = path.join(os.homedir(), 'nni-experiments', this.getExperimentId());
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

    public getExperimentId(): string {
        assert(this.initialized);

        return this.experimentId;
    }

    public getBasePort(): number {
        assert(this.initialized);

        return this.basePort;
    }

    public isNewExperiment(): boolean {
        assert(this.initialized);

        return this.newExperiment;
    }

    public getPlatform(): string {
        assert(this.initialized);

        return this.platform;
    }

    public getLogDir(): string {
        assert(this.initialized);

        return this.logDir;
    }

    public getLogLevel(): string {
        assert(this.initialized);

        return this.logLevel;
    }

    public isReadonly(): boolean {
        assert(this.initialized);

        return this.readonly;
    }

    public getDispatcherPipe(): string | null {
        assert(this.initialized);
        return this.dispatcherPipe;
    }

    public getAPIRootUrl(): string {
        assert(this.initialized);
        return this.urlprefix==''?this.API_ROOT_URL:`/${this.urlprefix}`;
    }
}

function getExperimentId(): string {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getExperimentId();
}

function getBasePort(): number {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getBasePort();
}

function isNewExperiment(): boolean {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).isNewExperiment();
}

function getPlatform(): string {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getPlatform();
}

function getExperimentStartupInfo(): ExperimentStartupInfo {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo);
}

function setExperimentStartupInfo(
    newExperiment: boolean, experimentId: string, basePort: number, platform: string, logDir?: string, logLevel?: string, readonly?: boolean, dispatcherPipe?: string, urlprefix?: string): void {
    component.get<ExperimentStartupInfo>(ExperimentStartupInfo)
        .setStartupInfo(newExperiment, experimentId, basePort, platform, logDir, logLevel, readonly, dispatcherPipe, urlprefix);
}

function isReadonly(): boolean {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).isReadonly();
}

function getDispatcherPipe(): string | null {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getDispatcherPipe();
}

function getAPIRootUrl(): string {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getAPIRootUrl();
}

export {
    ExperimentStartupInfo, getBasePort, getExperimentId, isNewExperiment, getPlatform, getExperimentStartupInfo,
    setExperimentStartupInfo, isReadonly, getDispatcherPipe, getAPIRootUrl
};
