// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import { MethodNotImplementedError, NNIError, NNIErrorNames } from '../../common/errors';
import { getLogger, Logger } from '../../common/log';
import { TrialJobStatus } from '../../common/trainingService';
import { KubernetesCRDClient } from './kubernetesApiClient';
import { KubernetesTrialJobDetail } from './kubernetesData';

/**
 * Collector Kubeflow jobs info from Kubernetes cluster, and update kubeflow job status locally
 */
export class KubernetesJobInfoCollector {
    protected readonly trialJobsMap: Map<string, KubernetesTrialJobDetail>;
    protected readonly log: Logger = getLogger();
    protected readonly statusesNeedToCheck: TrialJobStatus[];

    constructor(jobMap: Map<string, KubernetesTrialJobDetail>) {
        this.trialJobsMap = jobMap;
        this.statusesNeedToCheck = ['RUNNING', 'WAITING'];
    }

    public async retrieveTrialStatus(kubernetesCRDClient: KubernetesCRDClient | undefined): Promise<void[]> {
        assert(kubernetesCRDClient !== undefined);
        const updateKubernetesTrialJobs: Promise<void>[] = [];
        for (const [trialJobId, kubernetesTrialJob] of this.trialJobsMap) {
            if (kubernetesTrialJob === undefined) {
                throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
            }
            updateKubernetesTrialJobs.push(this.retrieveSingleTrialJobInfo(kubernetesCRDClient, kubernetesTrialJob));
        }
        return Promise.all(updateKubernetesTrialJobs);
    }

    protected async retrieveSingleTrialJobInfo(_kubernetesCRDClient: KubernetesCRDClient | undefined,
                                               _kubernetesTrialJob: KubernetesTrialJobDetail): Promise<void> {
            throw new MethodNotImplementedError();
    }
}
