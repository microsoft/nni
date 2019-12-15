// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as request from 'request';
import { Deferred } from 'ts-deferred';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getLogger, Logger } from '../../common/log';
import { TrialJobStatus } from '../../common/trainingService';
import { PAIBaseClusterConfig, PAIBaseTrialJobDetail } from './paiBaseConfig';

/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
export class PAIBaseJobInfoCollector {
    private readonly trialJobsMap: Map<string, PAIBaseTrialJobDetail>;
    private readonly log: Logger = getLogger();
    private readonly statusesNeedToCheck: TrialJobStatus[];
    private readonly finalStatuses: TrialJobStatus[];

    constructor(jobMap: Map<string, PAIBaseTrialJobDetail>) {
        this.trialJobsMap = jobMap;
        this.statusesNeedToCheck = ['RUNNING', 'UNKNOWN', 'WAITING'];
        this.finalStatuses = ['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED', 'EARLY_STOPPED'];
    }

    public async retrieveTrialStatus(token? : string, paiBaseClusterConfig?: PAIBaseClusterConfig): Promise<void> {
        if (paiBaseClusterConfig === undefined || token === undefined) {
            return Promise.resolve();
        }

        const updatePaiTrialJobs: Promise<void>[] = [];
        for (const [trialJobId, paiBaseTrialJob] of this.trialJobsMap) {
            if (paiBaseTrialJob === undefined) {
                throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
            }
            updatePaiTrialJobs.push(this.getSinglePAITrialJobInfo(paiBaseTrialJob, token, paiBaseClusterConfig));
        }

        await Promise.all(updatePaiTrialJobs);
    }

    private getSinglePAITrialJobInfo(paiBaseTrialJob: PAIBaseTrialJobDetail, paiToken: string, paiBaseClusterConfig: PAIBaseClusterConfig): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        if (!this.statusesNeedToCheck.includes(paiBaseTrialJob.status)) {
            deferred.resolve();

            return deferred.promise;
        }

        // Rest call to get PAI job info and update status
        // Refer https://github.com/Microsoft/pai/blob/master/docs/rest-server/API.md for more detail about PAI Rest API
        const getJobInfoRequest: request.Options = {
            uri: `http://${paiBaseClusterConfig.host}/rest-server/api/v1/user/${paiBaseClusterConfig.userName}/jobs/${paiBaseTrialJob.paiJobName}`,
            method: 'GET',
            json: true,
               headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${paiToken}`
            }
        };

        //TODO : pass in request timeout param?
        request(getJobInfoRequest, (error: Error, response: request.Response, body: any) => {
            if ((error !== undefined && error !== null) || response.statusCode >= 500) {
                this.log.error(`PAI Training service: get job info for trial ${paiBaseTrialJob.id} from PAI Cluster failed!`);
                // Queried PAI job info failed, set job status to UNKNOWN
                if (paiBaseTrialJob.status === 'WAITING' || paiBaseTrialJob.status === 'RUNNING') {
                    paiBaseTrialJob.status = 'UNKNOWN';
                }
            } else {
                if (response.body.jobStatus && response.body.jobStatus.state) {
                    switch (response.body.jobStatus.state) {
                        case 'WAITING':
                            paiBaseTrialJob.status = 'WAITING';
                            break;
                        case 'RUNNING':
                            paiBaseTrialJob.status = 'RUNNING';
                            if (paiBaseTrialJob.startTime === undefined) {
                                paiBaseTrialJob.startTime = response.body.jobStatus.appLaunchedTime;
                            }
                            if (paiBaseTrialJob.url === undefined) {
                                paiBaseTrialJob.url = response.body.jobStatus.appTrackingUrl;
                            }
                            break;
                        case 'SUCCEEDED':
                            paiBaseTrialJob.status = 'SUCCEEDED';
                            break;
                        case 'STOPPED':
                            if (paiBaseTrialJob.isEarlyStopped !== undefined) {
                                paiBaseTrialJob.status = paiBaseTrialJob.isEarlyStopped === true ?
                                        'EARLY_STOPPED' : 'USER_CANCELED';
                            } else {
                                /* if paiTrialJob's isEarlyStopped is undefined, that mean we didn't stop it via cancellation,
                                 * mark it as SYS_CANCELLED by PAI
                                 */
                                paiBaseTrialJob.status = 'SYS_CANCELED';
                            }
                            break;
                        case 'FAILED':
                            paiBaseTrialJob.status = 'FAILED';
                            break;
                        default:
                            paiBaseTrialJob.status = 'UNKNOWN';
                    }
                    // For final job statues, update startTime, endTime and url
                    if (this.finalStatuses.includes(paiBaseTrialJob.status)) {
                        if (paiBaseTrialJob.startTime === undefined) {
                            paiBaseTrialJob.startTime = response.body.jobStatus.appLaunchedTime;
                        }
                        if (paiBaseTrialJob.endTime === undefined) {
                            paiBaseTrialJob.endTime = response.body.jobStatus.completedTime;
                        }
                        // Set pai trial job's url to WebHDFS output path
                        if (paiBaseTrialJob.logPath !== undefined) {
                            paiBaseTrialJob.url += `,${paiBaseTrialJob.logPath}`;
                        }
                    }
                }
            }
            deferred.resolve();
        });

        return deferred.promise;
    }
}
