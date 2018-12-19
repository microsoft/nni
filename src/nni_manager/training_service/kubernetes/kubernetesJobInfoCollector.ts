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

import * as assert from 'assert';
import { getLogger, Logger } from '../../common/log';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { TrialJobStatus } from '../../common/trainingService';
import { KubernetesCRDClient } from './kubernetesApiClient';
import { MethodNotImplementedError } from '../../common/errors';
import { KubernetesTrialJobDetail } from './kubernetesData';

/**
 * Collector Kubeflow jobs info from Kubernetes cluster, and update kubeflow job status locally
 */
export class KubernetesJobInfoCollector {
    protected readonly trialJobsMap : Map<string, KubernetesTrialJobDetail>;
    protected readonly log: Logger = getLogger();
    protected readonly statusesNeedToCheck: TrialJobStatus[];

    constructor(jobMap: Map<string, KubernetesTrialJobDetail>) {
        this.trialJobsMap = jobMap;
        this.statusesNeedToCheck = ['RUNNING', 'WAITING'];
    }

    public async retrieveTrialStatus(kubernetesCRDClient: KubernetesCRDClient | undefined) : Promise<void> {
        assert(kubernetesCRDClient !== undefined);
        const updateKubernetesTrialJobs : Promise<void>[] = [];
        for(let [trialJobId, kubernetesTrialJob] of this.trialJobsMap) {
            if (!kubernetesTrialJob) {
                throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
            }
            // Since Kubeflow needs some delay to schedule jobs, we provide 20 seconds buffer time to check kubeflow job's status
            if( Date.now() - kubernetesTrialJob.submitTime < 20 * 1000) {
                return Promise.resolve();
            }
            updateKubernetesTrialJobs.push(this.retrieveSingleTrialJobInfo(kubernetesCRDClient, kubernetesTrialJob))
        }

        await Promise.all(updateKubernetesTrialJobs);
    }

    protected async retrieveSingleTrialJobInfo(kubernetesCRDClient: KubernetesCRDClient | undefined, 
        kubernetesTrialJob : KubernetesTrialJobDetail) : Promise<void> {
            throw new MethodNotImplementedError();
    }
}