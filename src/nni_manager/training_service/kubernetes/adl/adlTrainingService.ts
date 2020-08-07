// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as component from '../../../common/component';

import { String } from 'typescript-string-operations';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import {
    NNIManagerIpConfig, TrialJobApplicationForm, TrialJobDetail, TrialJobStatus
} from '../../../common/trainingService';
import { delay, generateParamFileName, getVersion, uniqueString } from '../../../common/utils';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
// import { validateCodeDir } from '../../common/util';
// import { NFSConfig } from '../kubernetesConfig';
import { KubernetesTrialJobDetail } from '../kubernetesData';
import { KubernetesTrainingService } from '../kubernetesTrainingService';
import { AdlClientFactory } from './adlApiClient'
import { AdlJobInfoCollector } from './adlJobInfoCollector';
import { AdlJobRestServer } from './adlJobRestServer';
import { AdlTrialConfig } from './adlConfig'

/**
 * Training Service implementation for Adl
 */
@component.Singleton
class AdlTrainingService extends KubernetesTrainingService implements KubernetesTrainingService {
    private adlTrialConfig?: AdlTrialConfig;
    private readonly adlJobInfoCollector: AdlJobInfoCollector;
    private configmapTemplate: any;
    private jobTemplate: any;
    private pvcTemplate: any;

    constructor() {
        super();
        this.adlJobInfoCollector = new AdlJobInfoCollector(this.trialJobsMap);
        this.experimentId = getExperimentId();
        this.kubernetesCRDClient = AdlClientFactory.createClient();
        this.configmapTemplate = JSON.parse(
            fs.readFileSync('./config/adl/adaptdl-nni-configmap-template.json', 'utf8'));
        this.jobTemplate = JSON.parse(
            fs.readFileSync('./config/adl/adaptdljob-template.json', 'utf8'));
        this.pvcTemplate = JSON.parse(
            fs.readFileSync('./config/adl/adaptdl-pvc-template.json', 'utf8'));
        this.log.info('Construct Adl training service.');
    }

    public async run(): Promise<void> {
        this.log.info('Run Adl training service.');
        this.kubernetesJobRestServer = component.get(AdlJobRestServer);
        if (this.kubernetesJobRestServer === undefined) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        await this.kubernetesJobRestServer.start();
        this.kubernetesJobRestServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`Adl Training service rest server listening on: ${this.kubernetesJobRestServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics for Adl jobs by interacting with Kubernetes API server
            await delay(3000);
            await this.adlJobInfoCollector.retrieveTrialStatus(this.kubernetesCRDClient);
            if (this.kubernetesJobRestServer.getErrorMessage !== undefined) {
                throw new Error(this.kubernetesJobRestServer.getErrorMessage);
            }
        }
        this.log.info('Adl training service exit.');
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Adl job operator client is undefined');
        }

        if (this.adlTrialConfig === undefined) {
            throw new Error('Adl trial config is undefined');
        }

        if (this.kubernetesRestServerPort === undefined) {
            const restServer: AdlJobRestServer = component.get(AdlJobRestServer);
            this.kubernetesRestServerPort = restServer.clusterRestServerPort;
        }

        const trialJobId: string = uniqueString(5);
        const adlJobName: string = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
        const initStatus: TrialJobStatus = 'WAITING';
        const codeDir = "."
        const outputDir = "output"
        const trialJobDetail: KubernetesTrialJobDetail = new KubernetesTrialJobDetail(
            trialJobId,
            initStatus,
            Date.now(),
            codeDir,
            form,
            adlJobName,
            outputDir
        );

        // Create adljob
        this.jobTemplate.metadata.name = adlJobName
        this.jobTemplate.metadata.labels.app = this.NNI_KUBERNETES_TRIAL_LABEL
        this.jobTemplate.metadata.labels.expId = this.experimentId
        this.jobTemplate.metadata.labels.trialId = trialJobId
        this.jobTemplate.spec.template.spec.containers[0]
            .image = this.adlTrialConfig.image;
        this.jobTemplate.spec.template.spec.containers[0]
            .resources.limits["nvidia.com/gpu"] = this.adlTrialConfig.gpuNum;
        this.jobTemplate.spec.template.spec.volumes[0]
            .persistentVolumeClaim.claimName = adlJobName
        this.jobTemplate.spec.template.spec.volumes[1]
            .configMap.name = adlJobName
        this.jobTemplate.spec.template.spec.imagePullSecrets = this.jobTemplate
            .spec.template.spec.imagePullSecrets.concat(
                this.adlTrialConfig.imagePullSecrets);
        await this.kubernetesCRDClient.createKubernetesJob(this.jobTemplate);
        const k8sadlJob: any = await this.kubernetesCRDClient.getKubernetesJob(adlJobName);

        // Create pvc
        this.pvcTemplate.metadata.name = adlJobName;
        this.pvcTemplate.metadata.ownerReferences[0].name = adlJobName;
        this.pvcTemplate.metadata.ownerReferences[0].uid = k8sadlJob.metadata.uid;
        this.pvcTemplate.spec.resources.requests.storage = this.adlTrialConfig
            .checkpoint.storageSize;
        this.pvcTemplate.spec.storageClassName = this.adlTrialConfig.checkpoint.storageClass;
        await this.genericK8sClient.createPersistentVolumeClaim(this.pvcTemplate);

        // prepare the runscript and convert it to configmap and mount it
        this.configmapTemplate.metadata.name = adlJobName;
        this.configmapTemplate.metadata.ownerReferences[0].name = adlJobName;
        this.configmapTemplate.metadata.ownerReferences[0].uid = k8sadlJob.metadata.uid;
        this.configmapTemplate.data["run.sh"] = await this.prepareRunSh(
            trialJobId, form, codeDir, outputDir)
        await this.genericK8sClient.createConfigMap(this.configmapTemplate)

        // Set trial job detail until create Adl job successfully
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    private async prepareRunSh(jobId: string,
                               form: TrialJobApplicationForm,
                               codeDir: string,
                               outputDir: string): Promise<string> {
        if (this.adlTrialConfig === undefined) {
            throw new Error('Adl trial config is undefined');
        }

        if (this.kubernetesRestServerPort === undefined) {
            throw new Error('Adl rest server port is undefined');
        }

        if (this.nniManagerIpConfig === undefined) {
            throw new Error('Adl nniManager ip config is undefined');
        }

        const expId: string = this.experimentId;
        const seqId: string = form.sequenceId.toString();
        const command: string = this.adlTrialConfig.command;
        const hyperParameters: string = form.hyperParameters.value;
        const hyperParametersFile: string = generateParamFileName(form.hyperParameters);
        const nniManagerPort: string = this.kubernetesRestServerPort.toString();
        const nniManagerIp: string = this.nniManagerIpConfig.nniManagerIp;
        let nniManagerVersion: string = '';
        if (this.versionCheck) {
            nniManagerVersion = await getVersion();
        }

        let nvidiaScript: string = '';
        if (this.adlTrialConfig.gpuNum == 0) {
            nvidiaScript = 'export CUDA_VISIBLE_DEVICES=';
        }

        const runScriptTemplate: string =
`#!/bin/bash
export NNI_PLATFORM=adl
export MULTI_PHASE=false
export NNI_SYS_DIR={0}
export NNI_CODE_DIR={0}
export NNI_OUTPUT_DIR={1}
export NNI_TRIAL_JOB_ID={2}
export NNI_EXP_ID={3}
export NNI_TRIAL_SEQ_ID={4}
mkdir -p $NNI_OUTPUT_DIR
{5}
echo '{6}' > $NNI_CODE_DIR/{7}
python3 -m nni_trial_tool.trial_keeper --trial_command '{8}' \
--nnimanager_ip {9} --nnimanager_port {10} \
--nni_manager_version '{11}' --log_collection '{12}'
`;
        const runScript = String.Format(
            runScriptTemplate, codeDir, outputDir,
            jobId, expId, seqId, nvidiaScript,
            hyperParameters, hyperParametersFile, command,
            nniManagerIp, nniManagerPort, nniManagerVersion,
            this.logCollection);
        return Promise.resolve(runScript);
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        this.log.info('SetCluster ' + key + ', ' +value);
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                this.adlTrialConfig = <AdlTrialConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                break;
            case TrialConfigMetadataKey.LOG_COLLECTION:
                this.logCollection = value;
                break;
            default:
        }

        return Promise.resolve();
    }

    public getClusterMetadata(key: string): Promise<string> {
        let result: string;
        switch (key) {
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (this.adlTrialConfig === undefined) {
                    return Promise.reject(`${key} is not set yet`);
                }

                result = JSON.stringify(this.adlTrialConfig);
                break;
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                if (this.nniManagerIpConfig === undefined) {
                    return Promise.reject(`${key} is not set yet`);
                }

                result = JSON.stringify(this.nniManagerIpConfig);
                break;
            default:
                return Promise.reject(`${key} not set`);
        }

        return Promise.resolve(result);
    }
}
export { AdlTrainingService };
