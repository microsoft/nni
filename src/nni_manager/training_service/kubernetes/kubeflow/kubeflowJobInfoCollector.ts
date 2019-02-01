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

import { KubernetesTrialJobDetail} from '../kubernetesData';
import { KubernetesCRDClient } from '../kubernetesApiClient';
import { KubernetesJobInfoCollector } from '../kubernetesJobInfoCollector';
import { KubeflowJobStatus } from './kubeflowConfig';

/**
 * Collector Kubeflow jobs info from Kubernetes cluster, and update kubeflow job status locally
 */
export class KubeflowJobInfoCollector extends KubernetesJobInfoCollector{
    constructor(jobMap: Map<string, KubernetesTrialJobDetail>) {
        super(jobMap);
    }

    protected async retrieveSingleTrialJobInfo(kubernetesCRDClient: KubernetesCRDClient | undefined, 
        kubernetesTrialJob : KubernetesTrialJobDetail) : Promise<void> {
        if (!this.statusesNeedToCheck.includes(kubernetesTrialJob.status)) {
            return Promise.resolve();
        }

        if(kubernetesCRDClient === undefined) {
            return Promise.reject('kubernetesCRDClient is undefined');
        }

        let kubernetesJobInfo: any;
        try {
            kubernetesJobInfo = await kubernetesCRDClient.getKubernetesJob(kubernetesTrialJob.kubernetesJobName);            
        } catch(error) {
            // Notice: it maynot be a 'real' error since cancel trial job can also cause getKubernetesJob failed. 
            this.log.error(`Get job ${kubernetesTrialJob.kubernetesJobName} info failed, error is ${error}`);
            //This is not treat as a error status
            return Promise.resolve();
        }

        if(kubernetesJobInfo.status && kubernetesJobInfo.status.conditions) {
            const latestCondition = kubernetesJobInfo.status.conditions[kubernetesJobInfo.status.conditions.length - 1];
            const tfJobType : KubeflowJobStatus = <KubeflowJobStatus>latestCondition.type;
            switch(tfJobType) {
                case 'Created':
                    kubernetesTrialJob.status = 'WAITING';
                    kubernetesTrialJob.startTime = Date.parse(<string>latestCondition.lastUpdateTime);                    
                    break; 
                case 'Running':
                    kubernetesTrialJob.status = 'RUNNING';
                    if(!kubernetesTrialJob.startTime) {
                        kubernetesTrialJob.startTime = Date.parse(<string>latestCondition.lastUpdateTime);
                    }
                    break;
                case 'Failed':
                    kubernetesTrialJob.status = 'FAILED';
                    kubernetesTrialJob.endTime = Date.parse(<string>latestCondition.lastUpdateTime);                    
                    break;
                case  'Succeeded':
                    kubernetesTrialJob.status = 'SUCCEEDED';
                    kubernetesTrialJob.endTime = Date.parse(<string>latestCondition.lastUpdateTime);                    
                    break;
                default:
                    break;
            }
        }
        return Promise.resolve();
    }
}