// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { AdlClientV1 } from './adlApiClient';
import { KubernetesTrialJobDetail} from '../kubernetesData';
import { KubernetesJobInfoCollector } from '../kubernetesJobInfoCollector';
import { AdlJobStatus } from './adlConfig';

/**
 * Collector Adl jobs info from Kubernetes cluster, and update adl job status locally
 */
export class AdlJobInfoCollector extends KubernetesJobInfoCollector {
    constructor(jobMap: Map<string, KubernetesTrialJobDetail>) {
        super(jobMap);
    }

    protected async retrieveSingleTrialJobInfo(kubernetesCRDClient: AdlClientV1 | undefined,
                                               kubernetesTrialJob: KubernetesTrialJobDetail): Promise<void> {
        if (!this.statusesNeedToCheck.includes(kubernetesTrialJob.status)) {
            return Promise.resolve();
        }

        if (kubernetesCRDClient === undefined) {
            return Promise.reject('kubernetesCRDClient is undefined');
        }

        let kubernetesJobInfo: any;
        let kubernetesPodsInfo: any;
        try {
            kubernetesJobInfo = await kubernetesCRDClient.getKubernetesJob(kubernetesTrialJob.kubernetesJobName);
            kubernetesPodsInfo = await kubernetesCRDClient.getKubernetesPods(kubernetesTrialJob.kubernetesJobName);
        } catch (error) {
            // Notice: it maynot be a 'real' error since cancel trial job can also cause getKubernetesJob failed.
            this.log.error(`Get job ${kubernetesTrialJob.kubernetesJobName} info failed, error is ${error}`);

            //This is not treat as a error status
            return Promise.resolve();
        }
        /* eslint-disable require-atomic-updates */
        if (kubernetesJobInfo.status) {
            const phase: AdlJobStatus = <AdlJobStatus>kubernetesJobInfo.status.phase
            switch (phase) {
                case 'Pending':
                case 'Starting':
                    if (kubernetesPodsInfo.items.length > 0){
                        const msg: string = kubernetesPodsInfo.items[0].status.containerStatuses[0].state.waiting.reason
                        if (msg == "ImagePullBackOff" || msg == "ErrImagePull") {
                            kubernetesTrialJob.status = 'FAILED';
                        }
                        else {
                            kubernetesTrialJob.status = 'WAITING';
                        }
                        kubernetesTrialJob.message = kubernetesPodsInfo.items
                            .map((pod: any) => JSON.stringify(pod.status.containerStatuses))
                            .join('\n');
                    }
                    else {
                        kubernetesTrialJob.status = 'WAITING';
                    }
                    kubernetesTrialJob.startTime = Date.parse(<string>kubernetesJobInfo.metadata.creationTimestamp);
                    break;
                case 'Running':
                case 'Stopping':
                    kubernetesTrialJob.status = 'RUNNING';
                    kubernetesTrialJob.message = undefined;  //TODO(Petuum)
                    if (kubernetesTrialJob.startTime === undefined) {
                        kubernetesTrialJob.startTime = Date.parse(<string>kubernetesJobInfo.metadata.creationTimestamp);
                    }
                    break;
                case 'Failed':
                    kubernetesTrialJob.status = 'FAILED';
                    kubernetesTrialJob.message = kubernetesJobInfo.status.message;
                    // undefined => NaN as endTime here
                    kubernetesTrialJob.endTime = Date.parse(<string>kubernetesJobInfo.status.completionTimestamp);
                    break;
                case  'Succeeded':
                    kubernetesTrialJob.status = 'SUCCEEDED';
                    kubernetesTrialJob.endTime = Date.parse(<string>kubernetesJobInfo.status.completionTimestamp);
                    kubernetesTrialJob.message = `Succeeded at ${kubernetesJobInfo.status.completionTimestamp}`
                    break;
                default:
            }
        }
        /* eslint-enable require-atomic-updates */

        return Promise.resolve();
    }
}
