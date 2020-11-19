// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { ExperimentStatus } from './manager';

abstract class ExpManager {
    public abstract getExperimentsInfo(): Promise<JSON>;
    public abstract setExperimentPath(newPath: string): void;
    public abstract setStatus(experimentId: string, status: ExperimentStatus): void;
}

export {ExpManager};
