// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Inject } from 'typescript-ioc';
import * as component from '../../common/component';
import { ClusterJobRestServer } from '../common/clusterJobRestServer';
import { KubernetesTrainingService } from './kubernetesTrainingService';

/**
 * Kubeflow Training service Rest server, provides rest API to support kubeflow job metrics update
 *
 */
@component.Singleton
export class KubernetesJobRestServer extends ClusterJobRestServer {
    @Inject
    private readonly kubernetesTrainingService? : KubernetesTrainingService;

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor(kubernetesTrainingService: KubernetesTrainingService) {
        super();
        this.kubernetesTrainingService = kubernetesTrainingService;
    }

    protected handleTrialMetrics(jobId: string, metrics: any[]): void {
        if (this.kubernetesTrainingService === undefined) {
            throw Error('kubernetesTrainingService not initialized!');
        }
        // Split metrics array into single metric, then emit
        // Warning: If not split metrics into single ones, the behavior will  be UNKNOWN
        for (const singleMetric of metrics) {
            this.kubernetesTrainingService.MetricsEmitter.emit('metric', {
                id : jobId,
                data : singleMetric
            });
        }
    }
}
