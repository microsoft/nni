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

import * as component from '../../common/component';
import { Inject } from 'typescript-ioc';
import { RemoteMachineTrainingService } from './remoteMachineTrainingService';
import { ClusterJobRestServer } from '../common/clusterJobRestServer'

/**
 * RemoteMachine Training service Rest server, provides rest RemoteMachine to support remotemachine job metrics update
 * 
 */
@component.Singleton
export class RemoteMachineJobRestServer extends ClusterJobRestServer{
    @Inject
    private readonly remoteMachineTrainingService : RemoteMachineTrainingService;

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super();
        this.remoteMachineTrainingService = component.get(RemoteMachineTrainingService);
    }

    protected handleTrialMetrics(jobId : string, metrics : any[]) : void {
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