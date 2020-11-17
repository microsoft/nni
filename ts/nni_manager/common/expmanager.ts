// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

abstract class ExpManager {
    public abstract getExperimentsInfo(): Promise<JSON>;
    public abstract setExperimentPath(newPath: string): void;
}

export {ExpManager};
