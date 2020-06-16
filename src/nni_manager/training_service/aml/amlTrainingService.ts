// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as path from 'path';
import * as request from 'request';
import * as component from '../../common/component';

import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import {
    NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay } from '../../common/utils';
import { AMLClusterConfig, AMLTrialJobDetail, AMLTrialConfig } from './amlConfig';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir, execCopydir } from '../common/util';
import {
    generateParamFileName,
    getIPV4Address, getVersion, uniqueString
} from '../../common/utils';
import { PythonShell } from 'python-shell';

/**
 * Training Service implementation for OpenPAI (Open Platform for AI)
 * Refer https://github.com/Microsoft/pai for more info about OpenPAI
 */
@component.Singleton
abstract class AMLTrainingService implements TrainingService {
    private readonly log!: Logger;
    private readonly metricsEmitter: EventEmitter;
    private readonly trialJobsMap: Map<string, AMLTrialJobDetail>;
    private readonly expRootDir: string;
    private amlClusterConfig?: AMLClusterConfig;
    private amlTrialConfig?: AMLTrialConfig;
    private readonly jobQueue: string[];
    private stopping: boolean = false;
    private readonly experimentId!: string;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    private versionCheck: boolean = true;
    private isMultiPhase: boolean = false;
    private nniVersion?: string;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, AMLTrialJobDetail>();
        this.jobQueue = [];
        this.expRootDir = path.join('/nni', 'experiments', getExperimentId());
        this.experimentId = getExperimentId();
        this.log.info('Construct aml training service.');
    }

    public async run(): Promise<void> {
        this.log.info('Run AML training service.');
        await Promise.all([
            this.statusCheckingLoop(),
            this.submitJobLoop()]);
        this.log.info('AML training service exit.');
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.AML_CLUSTER_CONFIG:
                this.amlClusterConfig = <AMLClusterConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                if (this.amlClusterConfig === undefined) {
                    this.log.error('aml cluster config is not initialized');
                    break;
                }
                this.amlTrialConfig = <AMLTrialConfig>JSON.parse(value);
                // Validate to make sure codeDir doesn't have too many files
                await validateCodeDir(this.amlTrialConfig.codeDir);
                break;
            }
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                this.nniVersion = this.versionCheck ? await getVersion() : '';
                break;
            case TrialConfigMetadataKey.MULTI_PHASE:
                this.isMultiPhase = (value === 'true' || value === 'True');
                break;
            default:
                //Reject for unknown keys
                this.log.error(`Uknown key: ${key}`);
        }
    }

    private async submitJobLoop(): Promise<void> {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length > 0) {
                const trialJobId: string = this.jobQueue[0];
                if (await this.submitTrialJobToAML(trialJobId)) {
                    // Remove trial job with trialJobId from job queue
                    this.jobQueue.shift();
                } else {
                    // Break the while loop since failed to submitJob
                    break;
                }
            }
            await delay(3000);
        }
    }

    // update trial parameters for multi-phase
    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: AMLTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }

        return trialJobDetail;
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];

        for (const key of this.trialJobsMap.keys()) {
            jobs.push(await this.getTrialJob(key));
        }

        return jobs;
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.amlClusterConfig === undefined) {
            throw new Error(`paiClusterConfig not initialized!`);
        }
        if (this.amlTrialConfig === undefined) {
            throw new Error(`paiTrialConfig not initialized!`);
        }

        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        const trialJobId: string = uniqueString(5);
        //TODO: use HDFS working folder instead
        const trialWorkingFolder: string = path.join(this.expRootDir, 'trials', trialJobId);
        const amlJobName: string = `nni_exp_${this.experimentId}_trial_${trialJobId}`;
        const logPath: string = "";
        const trialJobDetail: AMLTrialJobDetail = new AMLTrialJobDetail(
            trialJobId,
            'WAITING',
            amlJobName,
            Date.now(),
            trialWorkingFolder,
            form,
            logPath);

        this.trialJobsMap.set(trialJobId, trialJobDetail);
        this.jobQueue.push(trialJobId);

        return trialJobDetail;
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if (this.amlClusterConfig === undefined) {
            throw new Error('AML Cluster config is not initialized');
        }

        const amlTrialJob: AMLTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (amlTrialJob === undefined) {
            throw new Error(`trial job ${trialJobId} not found`);
        }

        return amlTrialJob;
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }

    public get isMultiPhaseJobSupported(): boolean {
        return true;
    }

    public cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail: AMLTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            return Promise.reject(new Error(`cancelTrialJob: trial job id ${trialJobId} not found`));
        }

        if (this.amlClusterConfig === undefined) {
            return Promise.reject(new Error('PAI Cluster config is not initialized'));
        }

        return Promise.resolve();
    }

    public getClusterMetadata(_key: string): Promise<string> {
        throw new Error('Not implemented!');
    }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping AML training service...');
        this.stopping = true;
    }

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    private async statusCheckingLoop(): Promise<void> {
        while (!this.stopping) {

            await delay(3000);
        }
    }

    private async submitTrialJobToAML(trialJobId: string): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        const trialJobDetail: AMLTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (trialJobDetail === undefined) {
            throw new Error(`Failed to find PAITrialJobDetail for job ${trialJobId}`);
        }

        if (this.amlClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.amlTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        let pyshell = new PythonShell('jobSubmission.py', {
            scriptPath: './config/aml',
            pythonOptions: ['-u'], // get print results in real-time
            args: [
                '--subscription_id', this.amlClusterConfig.subscriptionId,
                '--resource_group', this.amlClusterConfig.resourceGroup,
                '--workspace_name', this.amlClusterConfig.workspaceName,
                '--computer_target', this.amlTrialConfig.computerTarget,
                '--docker_image', this.amlTrialConfig.image,
                '--experiment_name', this.experimentId,
                '--code_dir', this.amlTrialConfig.codeDir,
                '--script', this.amlTrialConfig.script
              ]
        });
        pyshell.on('message', function (message) {
            // received a message sent from the Python script (a simple "print" statement)
            console.log(message);
        });
        // end the input stream and allow the process to exit
        pyshell.end(function (err,code,signal) {
            if (err) throw err;
            console.log('The exit code was: ' + code);
            console.log('The exit signal was: ' + signal);
            console.log('finished');
            deferred.resolve();
        });
        return deferred.promise;
    }
}

export { AMLTrainingService };
