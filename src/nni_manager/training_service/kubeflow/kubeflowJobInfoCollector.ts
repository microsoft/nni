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

import * as cpp from 'child-process-promise';
import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { getLogger, Logger } from '../../common/log';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { KubeflowTrialJobDetail, KubeflowTFJobType} from './kubeflowData';
import { TrialJobStatus } from '../../common/trainingService';

/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
export class KubeflowJobInfoCollector {
    private readonly trialJobsMap : Map<string, KubeflowTrialJobDetail>;
    private readonly log: Logger = getLogger();
    private readonly statusesNeedToCheck : TrialJobStatus[];
    private readonly finalStatuses : TrialJobStatus[];

    constructor(jobMap: Map<string, KubeflowTrialJobDetail>) {
        this.trialJobsMap = jobMap;
        this.statusesNeedToCheck = ['RUNNING', 'UNKNOWN', 'WAITING'];
        this.finalStatuses = ['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED'];
    }

    public async retrieveTrialStatus() : Promise<void> {
        const updatePaiTrialJobs : Promise<void>[] = [];
        for(let [trialJobId, kubeflowTrialJob] of this.trialJobsMap) {
            if (!kubeflowTrialJob) {
                throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
            }
            updatePaiTrialJobs.push(this.retrieveSingleTrialJobInfo(kubeflowTrialJob))
        }

        await Promise.all(updatePaiTrialJobs);
    }

    private async retrieveSingleTrialJobInfo(kubeflowTrialJob : KubeflowTrialJobDetail) : Promise<void> {
        if (!this.statusesNeedToCheck.includes(kubeflowTrialJob.status)) {
            return Promise.resolve();
        }

        const result = await cpp.exec(`kubectl get tfjobs ${kubeflowTrialJob.kubeflowJobName} -o json`);
        if(!result.stderr) {
            this.log.error(`Get tfjobs ${kubeflowTrialJob.kubeflowJobName} failed`);
            kubeflowTrialJob.status = 'UNKNOWN';
        }
        console.log(`Kubectl get tfjobs info is ${result.stdout}`);
        
        const kubeflowJobInfo = JSON.parse(result.stdout);
        if(kubeflowJobInfo.status && kubeflowJobInfo.status.conditions) {
            const latestCondition = kubeflowJobInfo.status.conditions[kubeflowJobInfo.status.conditions.length - 1];
            console.log(`tfjobs ${kubeflowTrialJob.kubeflowJobName} latest status is ${latestCondition.type}`);
            const tfJobType : KubeflowTFJobType = <KubeflowTFJobType>latestCondition.type;
            switch(tfJobType) {
                case 'Created':
                    kubeflowTrialJob.status = 'WAITING';
                    kubeflowTrialJob.startTime = Date.parse(<string>latestCondition.lastUpdateTime);                    
                    break; 
                case 'Running':
                    kubeflowTrialJob.status = 'RUNNING';
                    if(!kubeflowTrialJob.startTime) {
                        kubeflowTrialJob.startTime = Date.parse(<string>latestCondition.lastUpdateTime);
                    }
                    break;
                case 'Failed':
                    kubeflowTrialJob.status = 'FAILED';
                    kubeflowTrialJob.endTime = Date.parse(<string>latestCondition.lastUpdateTime);                    
                    break;
                case  'Succeeded':
                    kubeflowTrialJob.status = 'SUCCEEDED';
                    kubeflowTrialJob.endTime = Date.parse(<string>latestCondition.lastUpdateTime);                    
                    break;
                default:
                    break;
            }
        }

        return Promise.resolve();
    }
}