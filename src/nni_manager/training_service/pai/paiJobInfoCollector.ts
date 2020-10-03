// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as request from 'request';
import { Deferred } from 'ts-deferred';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getLogger, Logger } from '../../common/log';
import { TrialJobStatus } from '../../common/trainingService';
import { PAIClusterConfig, PAITrialJobDetail } from './paiConfig';

/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
export class PAIJobInfoCollector {
    private readonly trialJobsMap: Map<string, PAITrialJobDetail>;
    private readonly log: Logger = getLogger();
    private readonly statusesNeedToCheck: TrialJobStatus[];
    private readonly finalStatuses: TrialJobStatus[];

    constructor(jobMap: Map<string, PAITrialJobDetail>) {
        this.trialJobsMap = jobMap;
        this.statusesNeedToCheck = ['RUNNING', 'UNKNOWN', 'WAITING'];
        this.finalStatuses = ['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED', 'EARLY_STOPPED'];
    }

    public async retrieveTrialStatus(protocol: string, token? : string, paiBaseClusterConfig?: PAIClusterConfig): Promise<void> {
        if (paiBaseClusterConfig === undefined || token === undefined) {
            return Promise.resolve();
        }

        const updatePaiTrialJobs: Promise<void>[] = [];
        for (const [trialJobId, paiTrialJob] of this.trialJobsMap) {
            if (paiTrialJob === undefined) {
                throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
            }
            updatePaiTrialJobs.push(this.getSinglePAITrialJobInfo(protocol, paiTrialJob, token, paiBaseClusterConfig));
        }

        await Promise.all(updatePaiTrialJobs);
    }

    private getSinglePAITrialJobInfo(protocol: string, paiTrialJob: PAITrialJobDetail, paiToken: string, paiClusterConfig: PAIClusterConfig): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        if (!this.statusesNeedToCheck.includes(paiTrialJob.status)) {
            deferred.resolve();

            return deferred.promise;
        }

        // Rest call to get PAI job info and update status
        // Refer https://github.com/Microsoft/pai/blob/master/docs/rest-server/API.md for more detail about PAI Rest API
        const getJobInfoRequest: request.Options = {
            uri: `${protocol}://${paiClusterConfig.host}/rest-server/api/v2/jobs/${paiClusterConfig.userName}~${paiTrialJob.paiJobName}`,
            method: 'GET',
            json: true,
               headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${paiToken}`
            }
        };

        //TODO : pass in request timeout param?
        request(getJobInfoRequest, (error: Error, response: request.Response, _body: any) => {
            // Status code 200 for success
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                // The job refresh time could be ealier than job submission, so it might return 404 error code, need refactor
                // Queried PAI job info failed, set job status to UNKNOWN
                if (paiTrialJob.status === 'WAITING' || paiTrialJob.status === 'RUNNING') {
                    paiTrialJob.status = 'UNKNOWN';
                }
            } else {
                if (response.body.jobStatus && response.body.jobStatus.state) {
                    switch (response.body.jobStatus.state) {
                        case 'WAITING':
                            paiTrialJob.status = 'WAITING';
                            break;
                        case 'RUNNING':
                            paiTrialJob.status = 'RUNNING';
                            if (paiTrialJob.startTime === undefined) {
                                paiTrialJob.startTime = response.body.jobStatus.appLaunchedTime;
                            }
                            if (paiTrialJob.url === undefined) {
                                if (response.body.jobStatus.appTrackingUrl) {
                                    paiTrialJob.url = response.body.jobStatus.appTrackingUrl;
                                } else {
                                    paiTrialJob.url = paiTrialJob.paiJobDetailUrl;
                                }
                            }
                            break;
                        case 'SUCCEEDED':
                            paiTrialJob.status = 'SUCCEEDED';
                            break;
                        case 'STOPPED':
                        case 'STOPPING':
                            if (paiTrialJob.isEarlyStopped !== undefined) {
                                paiTrialJob.status = paiTrialJob.isEarlyStopped === true ?
                                        'EARLY_STOPPED' : 'USER_CANCELED';
                            } else {
                                /* if paiTrialJob's isEarlyStopped is undefined, that mean we didn't stop it via cancellation,
                                 * mark it as SYS_CANCELLED by PAI
                                 */
                                paiTrialJob.status = 'SYS_CANCELED';
                            }
                            break;
                        case 'FAILED':
                            paiTrialJob.status = 'FAILED';
                            break;
                        default:
                            paiTrialJob.status = 'UNKNOWN';
                    }
                    // For final job statues, update startTime, endTime and url
                    if (this.finalStatuses.includes(paiTrialJob.status)) {
                        if (paiTrialJob.startTime === undefined) {
                            paiTrialJob.startTime = response.body.jobStatus.appLaunchedTime;
                        }
                        if (paiTrialJob.endTime === undefined) {
                            paiTrialJob.endTime = response.body.jobStatus.completedTime;
                        }
                        // Set pai trial job's url to WebHDFS output path
                        if (paiTrialJob.logPath !== undefined) {
                            if (paiTrialJob.url && paiTrialJob.url !== paiTrialJob.logPath) {
                                paiTrialJob.url += `,${paiTrialJob.logPath}`;
                            } else {
                                paiTrialJob.url = `${paiTrialJob.logPath}`;
                            }
                        }
                    }
                }
            }
            deferred.resolve();
        });

        return deferred.promise;
    }
}
