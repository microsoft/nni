// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { KubernetesCRDClient } from '../kubernetesApiClient';
import { KubernetesTrialJobDetail} from '../kubernetesData';
import { KubernetesJobInfoCollector } from '../kubernetesJobInfoCollector';
import { KubeflowJobStatus } from './kubeflowConfig';

/**
 * Collector Kubeflow jobs info from Kubernetes cluster, and update kubeflow job status locally
 */
export class KubeflowJobInfoCollector extends KubernetesJobInfoCollector {
    constructor(jobMap: Map<string, KubernetesTrialJobDetail>) {
        super(jobMap);
    }

    protected async retrieveSingleTrialJobInfo(kubernetesCRDClient: KubernetesCRDClient | undefined,
                                               kubernetesTrialJob: KubernetesTrialJobDetail): Promise<void> {
        if (!this.statusesNeedToCheck.includes(kubernetesTrialJob.status)) {
            return Promise.resolve();
        }

        if (kubernetesCRDClient === undefined) {
            return Promise.reject('kubernetesCRDClient is undefined');
        }

        let kubernetesJobInfo: any;
        try {
            kubernetesJobInfo = await kubernetesCRDClient.getKubernetesJob(kubernetesTrialJob.kubernetesJobName);
        } catch (error) {
            // Notice: it maynot be a 'real' error since cancel trial job can also cause getKubernetesJob failed.
            this.log.error(`Get job ${kubernetesTrialJob.kubernetesJobName} info failed, error is ${error}`);

            //This is not treat as a error status
            return Promise.resolve();
        }
        /* eslint-disable require-atomic-updates */
        if (kubernetesJobInfo.status && kubernetesJobInfo.status.conditions) {
            const latestCondition: any = kubernetesJobInfo.status.conditions[kubernetesJobInfo.status.conditions.length - 1];
            const tfJobType: KubeflowJobStatus = <KubeflowJobStatus>latestCondition.type;
            switch (tfJobType) {
                case 'Created':
                    kubernetesTrialJob.status = 'WAITING';
                    kubernetesTrialJob.startTime = Date.parse(<string>latestCondition.lastUpdateTime);
                    break;
                case 'Running':
                    kubernetesTrialJob.status = 'RUNNING';
                    if (kubernetesTrialJob.startTime === undefined) {
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
            }
        }
        /* eslint-enable require-atomic-updates */

        return Promise.resolve();
    }
}
