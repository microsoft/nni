// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

abstract class ExperimentManager {
    public abstract getExperimentsInfo(): Promise<JSON>;
    public abstract setExperimentPath(newPath: string): void;
    public abstract setExperimentInfo(experimentId: string, key: string, value: any): void;
    public abstract stop(): Promise<void>;
}

export {ExperimentManager};
