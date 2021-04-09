// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { ClusterJobRestServer } from '../common/clusterJobRestServer';
import { RemoteMachineTrainingService } from './remoteMachineTrainingService';

/**
 * RemoteMachine Training service Rest server, provides rest RemoteMachine to support remotemachine job metrics update
 *
 */
export class RemoteMachineJobRestServer extends ClusterJobRestServer {
    private readonly remoteMachineTrainingService: RemoteMachineTrainingService;

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor(remoteMachineTrainingService: RemoteMachineTrainingService) {
        super();
        this.remoteMachineTrainingService = remoteMachineTrainingService;
    }

    protected handleTrialMetrics(jobId: string, metrics: any[]): void {
        // Split metrics array into single metric, then emit
        // Warning: If not split metrics into single ones, the behavior will be UNKNOWNls
        for (const singleMetric of metrics) {
            this.remoteMachineTrainingService.MetricsEmitter.emit('metric', {
                id : jobId,
                data : singleMetric
            });
        }
    }
}
