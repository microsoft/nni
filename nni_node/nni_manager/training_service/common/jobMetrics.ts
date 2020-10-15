// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobStatus  } from '../../common/trainingService';

/**
 * Trial job metrics class
 * Representing trial job metrics properties
 */
export class JobMetrics {
    public readonly jobId: string;
    public readonly metrics: string[];
    public readonly jobStatus: TrialJobStatus;
    public readonly endTimestamp: number;

    constructor(jobId: string, metrics: string[], jobStatus: TrialJobStatus, endTimestamp: number) {
        this.jobId = jobId;
        this.metrics = metrics;
        this.jobStatus = jobStatus;
        this.endTimestamp = endTimestamp;
    }
}
