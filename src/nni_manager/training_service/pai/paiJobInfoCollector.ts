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

import * as request from 'request';
import { Deferred } from 'ts-deferred';
import { getLogger, Logger } from '../../common/log';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { PAITrialJobDetail } from './paiData';
import { PAIClusterConfig } from './paiConfig';
import { TrialJobStatus } from '../../common/trainingService';

/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
export class PAIJobInfoCollector {
    private readonly trialJobsMap : Map<string, PAITrialJobDetail>;
    private readonly log: Logger = getLogger();
    private readonly statusesNeedToCheck : TrialJobStatus[];
    private readonly finalStatuses : TrialJobStatus[];

    constructor(jobMap: Map<string, PAITrialJobDetail>) {
        this.trialJobsMap = jobMap;
        this.statusesNeedToCheck = ['RUNNING', 'UNKNOWN', 'WAITING'];
        this.finalStatuses = ['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED', 'EARLY_STOPPED'];
    }

    public async retrieveTrialStatus(paiToken? : string, paiClusterConfig?: PAIClusterConfig) : Promise<void> {
        if (!paiClusterConfig || !paiToken) {
            return Promise.resolve();            
        }

        const updatePaiTrialJobs : Promise<void>[] = [];
        for(let [trialJobId, paiTrialJob] of this.trialJobsMap) {
            if (!paiTrialJob) {
                throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
            }
            updatePaiTrialJobs.push(this.getSinglePAITrialJobInfo(paiTrialJob, paiToken, paiClusterConfig))
        }

        await Promise.all(updatePaiTrialJobs);
    }

    private getSinglePAITrialJobInfo(paiTrialJob : PAITrialJobDetail, paiToken : string, paiClusterConfig: PAIClusterConfig) : Promise<void> {
        const deferred : Deferred<void> = new Deferred<void>();
        if (!this.statusesNeedToCheck.includes(paiTrialJob.status)) {
            deferred.resolve();
            return deferred.promise;
        }

        // Rest call to get PAI job info and update status
        // Refer https://github.com/Microsoft/pai/blob/master/docs/rest-server/API.md for more detail about PAI Rest API
        const getJobInfoRequest: request.Options = {
            uri: `http://${paiClusterConfig.host}/rest-server/api/v1/user/${paiClusterConfig.userName}/jobs/${paiTrialJob.paiJobName}`,
            method: 'GET',
            json: true,
            headers: {
                "Content-Type": "application/json",
                "Authorization": 'Bearer ' + paiToken
            }
        };
        //TODO : pass in request timeout param? 
        request(getJobInfoRequest, (error: Error, response: request.Response, body: any) => {
            if (error || response.statusCode >= 500) {
                this.log.error(`PAI Training service: get job info for trial ${paiTrialJob.id} from PAI Cluster failed!`);
                // Queried PAI job info failed, set job status to UNKNOWN
                if(paiTrialJob.status === 'WAITING' || paiTrialJob.status === 'RUNNING') {
                    paiTrialJob.status = 'UNKNOWN';
                }
            } else {
                if(response.body.jobStatus && response.body.jobStatus.state) {
                    switch(response.body.jobStatus.state) {
                        case 'WAITING': 
                            paiTrialJob.status = 'WAITING';
                            break;
                        case 'RUNNING':
                            paiTrialJob.status = 'RUNNING';
                            if(!paiTrialJob.startTime) {
                                paiTrialJob.startTime = response.body.jobStatus.appLaunchedTime;
                            }
                            if(!paiTrialJob.url) {
                                paiTrialJob.url = response.body.jobStatus.appTrackingUrl;    
                            }
                            break;
                        case 'SUCCEEDED':
                            paiTrialJob.status = 'SUCCEEDED';
                            break;
                        case 'STOPPED':
                            if (paiTrialJob.isEarlyStopped !== undefined) {
                                paiTrialJob.status = paiTrialJob.isEarlyStopped === true ? 
                                        'EARLY_STOPPED' : 'USER_CANCELED';
                            } else {
                                // if paiTrialJob's isEarlyStopped is undefined, that mean we didn't stop it via cancellation, mark it as SYS_CANCELLED by PAI
                                paiTrialJob.status = 'SYS_CANCELED';
                            }
                            break;
                        case 'FAILED':
                            paiTrialJob.status = 'FAILED';                            
                            break;
                        default:
                            paiTrialJob.status = 'UNKNOWN';
                            break;
                    }
                    // For final job statues, update startTime, endTime and url
                    if(this.finalStatuses.includes(paiTrialJob.status)) {
                        if(!paiTrialJob.startTime) {
                            paiTrialJob.startTime = response.body.jobStatus.appLaunchedTime;
                        }
                        if(!paiTrialJob.endTime) {
                            paiTrialJob.endTime = response.body.jobStatus.completedTime;
                        }
                        // Set pai trial job's url to WebHDFS output path
                        if(paiTrialJob.hdfsLogPath) {
                            paiTrialJob.url += `,${paiTrialJob.hdfsLogPath}`;
                        }
                    }
                }
            }
            deferred.resolve();
        });

        return deferred.promise;
    }
}
