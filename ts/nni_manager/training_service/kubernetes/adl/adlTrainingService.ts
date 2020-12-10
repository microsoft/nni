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
    private configmapTemplateStr: string;
    private jobTemplateStr: string;
    private pvcTemplateStr: string;
    private tensorboardPvcTemplate: any;
    private tensorboardDeploymentTemplate: any;
    //TODO: change the logic here when we want to support multiple tensorboard
    private tensorboardName: string = "adaptdl-tensorboard-" + getExperimentId().toLowerCase();

    constructor() {
        super();
        this.adlJobInfoCollector = new AdlJobInfoCollector(this.trialJobsMap);
        this.experimentId = getExperimentId();
        this.configmapTemplateStr = fs.readFileSync(
            './config/adl/adaptdl-nni-configmap-template.json', 'utf8');
        this.jobTemplateStr = fs.readFileSync('./config/adl/adaptdljob-template.json', 'utf8');
        this.pvcTemplateStr = fs.readFileSync('./config/adl/adaptdl-pvc-template.json', 'utf8');
        this.tensorboardPvcTemplate = JSON.parse(
            fs.readFileSync('./config/adl/adaptdl-tensorboard-pvc-template.json', 'utf8'));
        this.tensorboardDeploymentTemplate = JSON.parse(
            fs.readFileSync('./config/adl/adaptdl-tensorboard-deployment-template.json', 'utf8'));

        this.log.info('Construct Adl training service.');
    }

    public async run(): Promise<void> {
        this.log.info(this.tensorboardName);
        this.log.info('Start tensorboard deployment.');
        await this.launchTensorboard()

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
    private async launchTensorboard(): Promise<void> {
        // Start the tensorboard at the beginning of the experiment.
        if (this.adlTrialConfig === undefined) {
            throw new Error('Adl trial config is undefined');
        }
        // Create tensorboard deployment
        this.tensorboardDeploymentTemplate.metadata.name = this.tensorboardName
        this.tensorboardDeploymentTemplate.metadata.labels.expId = this.experimentId
        this.tensorboardDeploymentTemplate.spec.selector.matchLabels.app = this.tensorboardName
        this.tensorboardDeploymentTemplate.spec.template.metadata.labels.app = this.tensorboardName
        this.tensorboardDeploymentTemplate.spec.template.spec.volumes[0]
            .persistentVolumeClaim.claimName = this.tensorboardName
        const deploymentUid: string = await this.genericK8sClient.createDeployment(this.tensorboardDeploymentTemplate);
        // Create pvc
        this.tensorboardPvcTemplate.metadata.name = this.tensorboardName;
        this.tensorboardPvcTemplate.metadata.ownerReferences[0].name = this.tensorboardName;
        this.tensorboardPvcTemplate.metadata.ownerReferences[0].uid = deploymentUid
        if (this.adlTrialConfig.checkpoint != undefined) {
            this.tensorboardPvcTemplate.spec.resources.requests.storage = this.adlTrialConfig.checkpoint.storageSize;
            this.tensorboardPvcTemplate.spec.storageClassName = this.adlTrialConfig.checkpoint.storageClass;
        }
        else {
            this.tensorboardPvcTemplate.spec.resources.requests.storage = "1Gi"
            this.tensorboardPvcTemplate.spec.storageClassName = await this.genericK8sClient.getStorageClass();
        }
        await this.genericK8sClient.createPersistentVolumeClaim(this.tensorboardPvcTemplate);

        return Promise.resolve()
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
        const codeDir = this.adlTrialConfig.codeDir;
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
        const job: any = JSON.parse(this.jobTemplateStr);
        job.metadata.name = adlJobName
        job.metadata.labels.app = this.NNI_KUBERNETES_TRIAL_LABEL
        job.metadata.labels.expId = this.experimentId
        job.metadata.labels.trialId = trialJobId
        if (this.adlTrialConfig.adaptive !== undefined){
            job.spec.preemptible = this.adlTrialConfig.adaptive
        }
        job.spec.template.spec.containers[0]
            .image = this.adlTrialConfig.image;
        job.spec.template.spec.volumes[0]
            .persistentVolumeClaim.claimName = adlJobName
        job.spec.template.spec.volumes[1]
            .persistentVolumeClaim.claimName = this.tensorboardName
        job.spec.template.spec.volumes[2]
            .configMap.name = adlJobName
        // Handle Pod Resource
        let cpu: number = 1;
        let memory: string = "1Gi";
        if (this.adlTrialConfig.cpuNum !== undefined) {
            cpu = this.adlTrialConfig.cpuNum;
        }
        if (this.adlTrialConfig.memorySize !== undefined) {
            memory = this.adlTrialConfig.memorySize;
        }
        job.spec.template.spec.containers[0]
            .resources.requests.memory = memory;
        job.spec.template.spec.containers[0]
            .resources.requests.cpu = cpu;
        job.spec.template.spec.containers[0]
            .resources.limits["nvidia.com/gpu"] = this.adlTrialConfig.gpuNum;
        // Handle imagePullSecrets
        if (this.adlTrialConfig.imagePullSecrets !== undefined) {
            job.spec.template.spec.imagePullSecrets = job.spec.template.spec
                .imagePullSecrets.concat(this.adlTrialConfig.imagePullSecrets);
        }
        // Handle NFS
        if (this.adlTrialConfig.nfs !== undefined) {
            job.spec.template.spec.volumes.push({
                "name": "nfs",
                "nfs": {
                    "server": this.adlTrialConfig.nfs.server,
                    "path": this.adlTrialConfig.nfs.path,
                    "readOnly": false
                }
            });
            job.spec.template.spec.containers[0].volumeMounts.push({
                "name": "nfs",
                "mountPath": this.adlTrialConfig.nfs.containerMountPath
            });
        }
        await this.kubernetesCRDClient.createKubernetesJob(job);
        const k8sadlJob: any = await this.kubernetesCRDClient.getKubernetesJob(adlJobName);

        // Create pvc
        const pvc: any = JSON.parse(this.pvcTemplateStr);
        pvc.metadata.name = adlJobName;
        pvc.metadata.ownerReferences[0].name = adlJobName;
        pvc.metadata.ownerReferences[0].uid = k8sadlJob.metadata.uid;
        if (this.adlTrialConfig.checkpoint != undefined) {
            pvc.spec.resources.requests.storage = this.adlTrialConfig
                .checkpoint.storageSize;
            pvc.spec.storageClassName = this.adlTrialConfig.checkpoint.storageClass;
        }
        else {
            pvc.spec.resources.requests.storage = "1Gi"
            pvc.spec.storageClassName = await this.genericK8sClient.getStorageClass();
        }
        await this.genericK8sClient.createPersistentVolumeClaim(pvc);

        // prepare the runscript and convert it to configmap and mount it
        const configmap: any = JSON.parse(this.configmapTemplateStr);
        configmap.metadata.name = adlJobName;
        configmap.metadata.ownerReferences[0].name = adlJobName;
        configmap.metadata.ownerReferences[0].uid = k8sadlJob.metadata.uid;
        configmap.data["run.sh"] = await this.prepareRunScript(
            trialJobId, form, codeDir, outputDir)
        const cleanupScriptTemplate: string =
`#!/bin/bash
ps aux | grep "python3 -m nni.tools.trial_tool.trial_keeper" | awk '{print $2}' | xargs kill -2
while true;
do
    proc=\`ps aux | grep "python3 -m nni.tools.trial_tool.trial_keeper" | awk '{print $2}' | grep "" -c\`
    if (( $proc == 1  )); then
        exit 0
    else
        echo "waiting"
    fi
    sleep 1
done
`;
        configmap.data["cleanup.sh"] = cleanupScriptTemplate
        await this.genericK8sClient.createConfigMap(configmap)

        // Set trial job detail until create Adl job successfully
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    private async prepareRunScript(jobId: string,
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
python3 -m nni.tools.trial_tool.trial_keeper --trial_command '{8}' \
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

    public async cleanUp(): Promise<void> {
        super.cleanUp();

        // Delete Tensorboard deployment
        try {
            await this.genericK8sClient.deleteDeployment("adaptdl-tensorboard-" + this.experimentId.toLowerCase());
            this.log.info('tensorboard deployment deleted');
        } catch (error) {
            this.log.error(`tensorboard deployment deletion failed: ${error.message}`);
        }
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        this.log.info('SetCluster ' + key + ', ' +value);
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                this.adlTrialConfig = <AdlTrialConfig>JSON.parse(value);
                let namespace: string = 'default';
                if (this.adlTrialConfig.namespace !== undefined) {
                    namespace = this.adlTrialConfig.namespace;
                }
                this.genericK8sClient.setNamespace = namespace;
                this.kubernetesCRDClient = AdlClientFactory.createClient(namespace);
                break;
            }
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
