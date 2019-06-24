/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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

    // tslint:disable-next-line:no-any
    protected handleTrialMetrics(jobId : string, metrics : any[]) : void {
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
