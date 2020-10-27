// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { KubernetesCRDClient } from '../kubernetesApiClient';
import { KubernetesTrialJobDetail} from '../kubernetesData';
import { KubernetesJobInfoCollector } from '../kubernetesJobInfoCollector';
import { FrameworkControllerJobCompleteStatus, FrameworkControllerJobStatus } from './frameworkcontrollerConfig';

/**
 * Collector frameworkcontroller jobs info from Kubernetes cluster, and update frameworkcontroller job status locally
 */
export class FrameworkControllerJobInfoCollector extends KubernetesJobInfoCollector {
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
            this.log.error(`Get job ${kubernetesTrialJob.kubernetesJobName} info failed, error is ${error}`);
            //This is not treat as a error status

            return Promise.resolve();
        }

        if (kubernetesJobInfo.status && kubernetesJobInfo.status.state) {
            const frameworkJobType: FrameworkControllerJobStatus = <FrameworkControllerJobStatus>kubernetesJobInfo.status.state;
            /* eslint-disable require-atomic-updates */
            switch (frameworkJobType) {
                case 'AttemptCreationPending':
                case 'AttemptCreationRequested':
                case 'AttemptPreparing':
                    kubernetesTrialJob.status = 'WAITING';
                    break;
                case 'AttemptRunning':
                    kubernetesTrialJob.status = 'RUNNING';
                    if (kubernetesTrialJob.startTime === undefined) {
                        kubernetesTrialJob.startTime = Date.parse(<string>kubernetesJobInfo.status.startTime);
                    }
                    break;
                case  'Completed': {
                    const completedJobType: FrameworkControllerJobCompleteStatus =
                      <FrameworkControllerJobCompleteStatus>kubernetesJobInfo.status.attemptStatus.completionStatus.type.name;
                    switch (completedJobType) {
                        case 'Succeeded':
                            kubernetesTrialJob.status = 'SUCCEEDED';
                            break;
                        case 'Failed':
                            kubernetesTrialJob.status = 'FAILED';
                            break;
                        default:
                    }
                    kubernetesTrialJob.endTime = Date.parse(<string>kubernetesJobInfo.status.completionTime);
                    break;
                }
                default:
            }
            /* eslint-enable require-atomic-updates */
        }

        return Promise.resolve();
    }
}
