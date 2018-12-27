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
import { FrameworkControllerJobStatus, FrameworkControllerJobCompleteStatus } from './frameworkcontrollerConfig';

/**
 * Collector frameworkcontroller jobs info from Kubernetes cluster, and update frameworkcontroller job status locally
 */
export class FrameworkControllerJobInfoCollector extends KubernetesJobInfoCollector{
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
            this.log.error(`Get job ${kubernetesTrialJob.kubernetesJobName} info failed, error is ${error}`);
            //This is not treat as a error status
            return Promise.resolve();
        }

        if(kubernetesJobInfo.status && kubernetesJobInfo.status.state) {
            const frameworkJobType: FrameworkControllerJobStatus = <FrameworkControllerJobStatus>kubernetesJobInfo.status.state;
            switch(frameworkJobType) {
                case 'AttemptCreationPending' || 'AttemptCreationRequested' || 'AttemptPreparing':
                    kubernetesTrialJob.status = 'WAITING';
                    break;
                case 'AttemptRunning':
                    kubernetesTrialJob.status = 'RUNNING';
                    if(!kubernetesTrialJob.startTime) {
                        kubernetesTrialJob.startTime = Date.parse(<string>kubernetesJobInfo.status.startTime);
                    }
                    break;
                case  'Completed':
                    const completedJobType : FrameworkControllerJobCompleteStatus = <FrameworkControllerJobCompleteStatus>kubernetesJobInfo.status.attemptStatus.completionStatus.type.name;
                    switch(completedJobType) {
                        case 'Succeeded':
                            kubernetesTrialJob.status = 'SUCCEEDED';
                            break;
                        case 'Failed':
                            kubernetesTrialJob.status = 'FAILED';
                            break;        
                    }
                    kubernetesTrialJob.endTime = Date.parse(<string>kubernetesJobInfo.status.completionTime); 
                    break;
                default:
                    break;
            }
        }
        return Promise.resolve();
    }
}