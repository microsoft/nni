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

import * as assert from 'assert';
import { EventEmitter } from 'events';
import * as path from 'path';
import { Client } from 'ssh2';
import { getLogger, Logger } from '../../common/log';
import { TrialJobStatus, TrialJobDetail } from '../../common/trainingService';
import { JobMetrics } from '../common/jobMetrics';
import { RemoteCommandResult, RemoteMachineMeta, RemoteMachineTrialJobDetail } from './remoteMachineData';
import { SSHClientUtility } from './sshClientUtility';

export class MetricsCollector {
    private machineSSHClientMap : Map<RemoteMachineMeta, Client>;
    private trialJobsMap : Map<string, any>;
    private expRootDir: string;
    private metricsEmitter: EventEmitter;
    private log: Logger = getLogger();

    constructor(clientMap: Map<RemoteMachineMeta, Client>,
                jobMap: Map<string, any>,
                expDir: string, eventEmitter: EventEmitter) {
        this.machineSSHClientMap = clientMap;
        this.trialJobsMap = jobMap;
        this.expRootDir = expDir;
        this.metricsEmitter = eventEmitter;
    }

    public async collectMetrics(): Promise<void> {
        const aliveJobStatus : TrialJobStatus[] = ['RUNNING', 'SUCCEEDED'];
        const runningJobsMap: Map<RemoteMachineMeta, string[]> = this.getTrialJobIdsGroupByRmMeta(aliveJobStatus);
        const readMetricsTasks: Promise<JobMetrics[]>[] = [];;
        runningJobsMap.forEach((jobIds: string[], rmMeta: RemoteMachineMeta) => {
            readMetricsTasks.push(this.readRmMetrics(rmMeta, jobIds));
        });
        const allMetrics = await Promise.all(readMetricsTasks.map(task => { return task.catch(err => { this.log.error(err.message); }); }));
        allMetrics.forEach((rmMetrics) => {
            if (rmMetrics !== undefined && rmMetrics.length > 0) {
                rmMetrics.forEach((jobMetrics) => {
                    const trialJobId : string = jobMetrics.jobId;
                    const trialJobDetail : RemoteMachineTrialJobDetail = <RemoteMachineTrialJobDetail>this.trialJobsMap.get(trialJobId);
                    assert(trialJobDetail);
                    // If job status is not alive again, remove its GPU reservation
                    if(!['RUNNING'].includes(jobMetrics.jobStatus)) {
                        if (trialJobDetail.status !== 'EARLY_STOPPED') {
                            trialJobDetail.status = jobMetrics.jobStatus;
                        }
                        this.log.debug(`Set trialjob ${trialJobDetail.id} status to ${trialJobDetail.status}`);
                        runningJobsMap.forEach((jobIds: string[], rmMeta: RemoteMachineMeta) => {
                            // If remote machine has no GPU, gpuReservcation is not initialized, so check if it's undefined
                            if(rmMeta.gpuReservation !== undefined) {
                                rmMeta.gpuReservation.forEach((reserveTrialJobId : string, gpuIndex : number) => {
                                    if(reserveTrialJobId == trialJobId) {
                                        rmMeta.gpuReservation.delete(gpuIndex);
                                    }
                                });
                            }
                        });
                    }
                    this.sendMetricsToListeners(jobMetrics);
                });
            }
        });
    }

    private getTrialJobIdsGroupByRmMeta(status: TrialJobStatus[]): Map<RemoteMachineMeta, string[]> {
        const map: Map<RemoteMachineMeta, string[]> = new Map<RemoteMachineMeta, string[]>();
        this.trialJobsMap.forEach((trialJob, id) => {
            let reservedTrialJobIds : string[] = [];
            if(trialJob.rmMeta !== undefined 
              && trialJob.rmMeta.gpuReservation !== undefined) {
                reservedTrialJobIds = Array.from(trialJob.rmMeta.gpuReservation.values());
            }
            if (reservedTrialJobIds.includes(id) || status.includes(trialJob.status)) {
                if (map.has(trialJob.rmMeta)) {
                    const ids = map.get(trialJob.rmMeta);
                    if (ids !== undefined && !ids.includes(id)) {
                        ids.push(id);
                    }
                } else {
                    let initJobIds : string[] = [id];
                    
                    // If the remote machine has jobs reserve GPU, also put that jobs into list to get metrics data
                    if(trialJob.rmMeta.gpuReservation !== undefined) {
                        const concatJobIds : string[] = initJobIds.concat(reservedTrialJobIds);                        
                        initJobIds = concatJobIds.filter((item, pos) => concatJobIds.indexOf(item) === pos);
                    }

                    map.set(trialJob.rmMeta, initJobIds);
                }
            }
        });

        return map;
    }

    private sendMetricsToListeners(jobMetrics: JobMetrics): void {
        if (jobMetrics === undefined) {
            return;
        }
        const jobId: string = jobMetrics.jobId;
        jobMetrics.metrics.forEach((metric: string) => {
            if (metric.length > 0) {
                this.metricsEmitter.emit('metric', {
                    id : jobId,
                    data : metric
                });
            }
        });
    }

    private async readRmMetrics(rmMeta: RemoteMachineMeta, trialJobIds: string[]): Promise<JobMetrics[]> {
        if (trialJobIds === undefined || trialJobIds.length < 1) {
            return [];
        }
        const scriptFile: string = path.join(path.dirname(path.dirname(this.expRootDir)), 'scripts', 'metrics_reader.py');
        const cmdStr: string = `python3 ${scriptFile} --experiment_dir ${this.expRootDir} --trial_job_ids ${trialJobIds.join(',')}`;

        trialJobIds.forEach((id: string) => {
            const trialJob: RemoteMachineTrialJobDetail = this.trialJobsMap.get(id);
            assert(trialJob.rmMeta === rmMeta);
        });
        const sshClient: Client | undefined = this.machineSSHClientMap.get(rmMeta);
        if (sshClient === undefined) {
            throw new Error('SSHClient not found!');
        }

        const result: RemoteCommandResult = await SSHClientUtility.remoteExeCommand(cmdStr, sshClient);
        if (result.exitCode !== 0) {
            throw new Error(`Failed to read metrics data: ${result.stderr}`);
        } else {
            if (result.stdout !== undefined && result.stdout.length > 0) {
                return <JobMetrics[]>JSON.parse(result.stdout);
            } else {
                return [];
            }
        }
    }
}
