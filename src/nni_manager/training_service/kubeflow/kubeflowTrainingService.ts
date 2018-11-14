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

'use strict'

import * as component from '../../common/component';
import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';

import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { EventEmitter } from 'events';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors';
import { String } from 'typescript-string-operations';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import {
    JobApplicationForm, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, getIPV4Address, uniqueString } from '../../common/utils';
import { KubeflowClusterConfig, KubeflowTrialConfig } from './kubeflowConfig';
import { KubeflowTrialJobDetail, KUBEFLOW_RUN_SHELL_FORMAT } from './kubeflowData';
import { KubeflowJobRestServer } from './kubeflowJobRestServer';
import { KubeflowJobInfoCollector } from './kubeflowJobInfoCollector';

var yaml = require('node-yaml');

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class KubeflowTrainingService implements TrainingService {
    private readonly log!: Logger;
    private readonly metricsEmitter: EventEmitter;
    private readonly trialJobsMap: Map<string, KubeflowTrialJobDetail>;
    // TODO: experiment root dir in NFS
    //private readonly expRootDir: string;
    private stopping: boolean = false;
    private experimentId! : string;
    private trialSequenceId: number;
    private kubeflowClusterConfig?: KubeflowClusterConfig;
    private kubeflowTrialConfig?: KubeflowTrialConfig;
    private kubeflowJobInfoCollector: KubeflowJobInfoCollector;
    private kubeflowRestServerPort?: number;

    constructor() {        
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, KubeflowTrialJobDetail>();
        this.kubeflowJobInfoCollector = new KubeflowJobInfoCollector(this.trialJobsMap);
        // TODO: Root dir on NFS
        //this.expRootDir = path.join('/nni', 'experiments', getExperimentId());
        this.experimentId = getExperimentId();      
        this.trialSequenceId = 0;
    }

    public async run(): Promise<void> {
        const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
        await restServer.start();
        this.log.info(`Kubeflow Training service rest server listening on: ${restServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics by calling 'kubectl get' command on Kubeflow jobs 
            await delay(3000);
            await this.kubeflowJobInfoCollector.retrieveTrialStatus();            
        }
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        if(!this.kubeflowRestServerPort) {
            const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
            this.kubeflowRestServerPort = restServer.clusterRestServerPort;
        }

        const trialJobId: string = uniqueString(5);
        const trialSequenceId: number = this.generateSequenceId();
        //TODO: use NFS working folder instead        
        const trialWorkingFolder: string = '';
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${path.dirname(trialLocalTempFolder)}`);
        await cpp.exec(`cp -r ${this.kubeflowTrialConfig.codeDir} ${trialLocalTempFolder}`);
        const trialLocalNFSTempFolder: string = path.join(getExperimentRootDir(), 'trials-nfs-tmp');
        await cpp.exec(`mkdir -p ${trialLocalNFSTempFolder}`);
        try {
            await cpp.exec(`sudo mount ${this.kubeflowClusterConfig.nfs.server}:${this.kubeflowClusterConfig.nfs.path} ${trialLocalNFSTempFolder}`);
        } catch(error) {
            this.log.error(`Mount NFS ${this.kubeflowClusterConfig.nfs.server}:${this.kubeflowClusterConfig.nfs.path} to ${trialLocalNFSTempFolder} failed, error is ${error}`);
        }

        const runScriptContent : string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        const kubeflowRunScriptContent: string = String.Format(
            KUBEFLOW_RUN_SHELL_FORMAT,
            `$PWD/nni/${trialJobId}`,
            `$PWD/nni/${trialJobId}`,
            trialJobId,
            getExperimentId(),
            //TODO: Remove hard-coded /tmp/nni?
            `/tmp/nni/${trialJobId}`,
            this.kubeflowTrialConfig.command,
            getIPV4Address(),
            this.kubeflowRestServerPort
            );            

        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${path.join(trialLocalTempFolder, '.nni')}`);

        // Write file content ( run.sh and parameter.cfg ) to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run.sh'), kubeflowRunScriptContent, { encoding: 'utf8' });

        // Write file content ( parameter.cfg ) to local tmp folders
        const trialForm : TrialJobApplicationForm = (<TrialJobApplicationForm>form)
        if(trialForm && trialForm.hyperParameters) {
            await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(trialForm.hyperParameters)), 
                            trialForm.hyperParameters.value, { encoding: 'utf8' });
            await fs.promises.writeFile(path.join(trialLocalTempFolder, '.nni', 'sequence_id'), trialSequenceId.toString(), { encoding: 'utf8' });
        }               

        const kubeflowJobYamlPath = path.join(trialLocalTempFolder, `kubeflow-job-${trialJobId}.yaml`);
        console.log(`kubeflow yaml config path is ${kubeflowJobYamlPath}`);
        const kubeflowJobName = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
        const podResources : any = {};
        podResources.requests = {
            'memory': `${this.kubeflowTrialConfig.memoryMB}Mi`,
            'cpu': `${this.kubeflowTrialConfig.cpuNum}`,
            'nvidia.com/gpu': `${this.kubeflowTrialConfig.gpuNum}`
        }

        podResources.limits = Object.assign({}, podResources.requests);

        // Generate tfjobs resource yaml file for K8S
        yaml.write(
            kubeflowJobYamlPath,
            this.generateKubeflowJobConfig(trialJobId, kubeflowJobName, podResources),
            'utf-8'
        );

        //TODO: refactor
        await cpp.exec(`mkdir -p ${trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}`);
        await cpp.exec(`cp -rT ${trialLocalTempFolder}/* ${trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}/.`);

        const trialJobDetail: KubeflowTrialJobDetail = new KubeflowTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            trialWorkingFolder,
            form,
            kubeflowJobName,
            trialSequenceId
            );
        
        // Create kubeflow training jobs
        await cpp.exec(`kubectl create -f ${kubeflowJobYamlPath}`);

        // Set trial job detail until kubectl create resource successfully 
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    public updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public listTrialJobs(): Promise<TrialJobDetail[]> {
        throw new MethodNotImplementedError();
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        const kubeflowTrialJob: TrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (!kubeflowTrialJob) {
            return Promise.reject(`trial job ${trialJobId} not found`)
        }        

        return Promise.resolve(kubeflowTrialJob);
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void) {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void) {
        this.metricsEmitter.off('metric', listener);
    }
 
    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public cancelTrialJob(trialJobId: string): Promise<void> {
        throw new MethodNotImplementedError();
    }

    public setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.KUBEFLOW_CLUSTER_CONFIG:
                this.kubeflowClusterConfig = <KubeflowClusterConfig>JSON.parse(value);
                console.log(`nfs server is ${this.kubeflowClusterConfig.nfs.server}`);
                console.log(`nfs path is ${this.kubeflowClusterConfig.nfs.path}`);
                //TODO: check NFS mount point here? 
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (!this.kubeflowClusterConfig){
                    this.log.error('kubeflow cluster config is not initialized');
                    return Promise.reject(new Error('kubeflow cluster config is not initialized'));                    
                }

                this.kubeflowTrialConfig = <KubeflowTrialConfig>JSON.parse(value);
                console.log(`kubeflow trial gpunumber is ${this.kubeflowTrialConfig.gpuNum}`);
                console.log(`kubeflow trial image is ${this.kubeflowTrialConfig.image}`);
                break;
            default:
                break;
        }

        return Promise.resolve();
    }

    public getClusterMetadata(key: string): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public cleanUp(): Promise<void> {
        this.stopping = true;
        //TODO: stop all running kubeflow jobs?? 
        return Promise.resolve();
    }

    public get MetricsEmitter() : EventEmitter {
        return this.metricsEmitter;
    }

    private generateKubeflowJobConfig(trialJobId: string, kubeflowJobName : string, podResources : any) : any {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        return {
            apiVersion: 'kubeflow.org/v1alpha2',
            kind: 'TFJob',
            metadata: { 
                // TODO: use random number for NNI pod name
                name: kubeflowJobName,
                namespace: 'default',
                labels: {
                    app: 'nni-kubeflow-trial'
                }
            },
            spec: {
                tfReplicaSpecs: {
                    Worker: {
                        replicas: 1,
                        template: {
                            metadata: {
                                creationTimestamp: null
                            },
                            spec: {
                                containers: [
                                {
                                    // Kubeflow tensorflow operator requires that containers' name must be tensorflow
                                    // TODO: change the name based on operator's type
                                    name: 'tensorflow',
                                    image: this.kubeflowTrialConfig.image,
                                    //TODO: change to real code dir
                                    args: ["sh", "/tmp/nni/nuDEP/run.sh"],                                    
                                    volumeMounts: [{
                                        name: 'nni-nfs-vol',
                                        mountPath: '/tmp/nni'
                                    }],
                                    resources: podResources//,
                                    //workingDir: '/tmp/nni/nuDEP'
                                }],
                                restartPolicy: 'ExitCode',
                                volumes: [{
                                    name: 'nni-nfs-vol',
                                    nfs: {
                                        server: `${this.kubeflowClusterConfig.nfs.server}`,
                                        path: `${this.kubeflowClusterConfig.nfs.path}`
                                    }
                                }]
                            }
                        }
                    }
                }
            }                
        };        
    }

    private generateSequenceId(): number {
        return this.trialSequenceId++;
    }
}

export { KubeflowTrainingService }