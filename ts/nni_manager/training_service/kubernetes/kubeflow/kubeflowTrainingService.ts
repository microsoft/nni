// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../common/component';

import { getExperimentId } from '../../../common/experimentStartupInfo';
import {
    TrialJobApplicationForm, TrialJobDetail, TrialJobStatus
} from '../../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, uniqueString } from '../../../common/utils';
import { ExperimentConfig, KubeflowConfig, flattenConfig, toMegaBytes } from '../../../common/experimentConfig';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { validateCodeDir } from '../../common/util';
import { NFSConfig } from '../kubernetesConfig';
import { KubernetesTrialJobDetail } from '../kubernetesData';
import { KubernetesTrainingService } from '../kubernetesTrainingService';
import { KubeflowOperatorClientFactory } from './kubeflowApiClient';
import { KubeflowClusterConfig, KubeflowClusterConfigAzure, KubeflowClusterConfigFactory, KubeflowClusterConfigNFS,
    KubeflowTrialConfig, KubeflowTrialConfigFactory, KubeflowTrialConfigPytorch, KubeflowTrialConfigTensorflow
} from './kubeflowConfig';
import { KubeflowJobInfoCollector } from './kubeflowJobInfoCollector';
import { KubeflowJobRestServer } from './kubeflowJobRestServer';

interface FlattenKubeflowConfig extends ExperimentConfig, KubeflowConfig { }

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class KubeflowTrainingService extends KubernetesTrainingService implements KubernetesTrainingService {
    private readonly kubeflowJobInfoCollector: KubeflowJobInfoCollector;
    private config: FlattenKubeflowConfig;
    private initializePromise: Promise<void> | undefined;

    constructor(config: ExperimentConfig) {
        super();
        this.kubeflowJobInfoCollector = new KubeflowJobInfoCollector(this.trialJobsMap);
        this.experimentId = getExperimentId();
        this.log.info('Construct Kubeflow training service.');
        this.config = flattenConfig(config, 'kubeflow');

        let storagePromise: Promise<void>;
        if (this.config.storage.storage === 'azureStorage') {
            this.azureStorageAccountName = this.config.storage.azureAccount;
            this.azureStorageShare = this.config.storage.azureShare;
            storagePromise = this.createAzureStorage(this.config.storage.keyVault!, this.config.storage.keyVaultSecret!);
        } else {
            storagePromise = this.createNFSStorage(this.config.storage.server!, this.config.storage.path!);
        }

        this.kubernetesCRDClient = KubeflowOperatorClientFactory.createClient(this.config.operator, this.config.apiVersion);

        const validatePromise = validateCodeDir(this.config.trialCodeDirectory);

        this.initializePromise = Promise.all([storagePromise, validatePromise]).then(() => {
            this.uploadFolder(this.config.trialCodeDirectory, `nni/${getExperimentId()}/nni-code`);
        });

        this.kubernetesJobRestServer = new KubeflowJobRestServer(this);
        this.kubernetesJobRestServer.setEnableVersionCheck = this.versionCheck;
    }

    public async run(): Promise<void> {
        this.log.debug('Checking if Kubeflow training service initialized...');
        await this.waitInitialized();
        this.log.info('Run Kubeflow training service.');
        await this.kubernetesJobRestServer.start();
        this.log.info(`Kubeflow Training service rest server listening on: ${this.kubernetesJobRestServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics for Kubeflow jobs by interacting with Kubernetes API server
            await delay(3000);
            await this.kubeflowJobInfoCollector.retrieveTrialStatus(this.kubernetesCRDClient);
            if (this.kubernetesJobRestServer.getErrorMessage !== undefined) {
                throw new Error(this.kubernetesJobRestServer.getErrorMessage);
                this.stopping = true;
            }
        }
        this.log.info('Kubeflow training service exit.');
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        await this.waitInitialized();

        if (this.kubernetesRestServerPort === undefined) {
            const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
            this.kubernetesRestServerPort = restServer.clusterRestServerPort;
        }

        // upload code Dir to storage
        if (this.copyExpCodeDirPromise !== undefined) {
            await this.copyExpCodeDirPromise;
        }

        const trialJobId: string = uniqueString(5);
        const trialWorkingFolder: string = path.join(this.CONTAINER_MOUNT_PATH, 'nni', getExperimentId(), trialJobId);
        const kubeflowJobName: string = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //prepare the runscript
        await this.prepareRunScript(trialLocalTempFolder, trialJobId, trialWorkingFolder, form);
        //upload script files to sotrage
        const trialJobOutputUrl: string = await this.uploadFolder(trialLocalTempFolder, `nni/${getExperimentId()}/${trialJobId}`);
        let initStatus: TrialJobStatus = 'WAITING';
        if (!trialJobOutputUrl) {
            initStatus = 'FAILED';
        }
        const trialJobDetail: KubernetesTrialJobDetail = new KubernetesTrialJobDetail(
            trialJobId,
            initStatus,
            Date.now(),
            trialWorkingFolder,
            form,
            kubeflowJobName,
            trialJobOutputUrl
        );

        // Generate kubeflow job resource config object
        const kubeflowJobConfig: any = await this.prepareKubeflowConfig(trialJobId, trialWorkingFolder, kubeflowJobName);
        // Create kubeflow job based on generated kubeflow job resource config
        await this.kubernetesCRDClient.createKubernetesJob(kubeflowJobConfig);

        // Set trial job detail until create Kubeflow job successfully
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    public async setClusterMetadata(key_: string, value_: string): Promise<void> { }
    public async getClusterMetadata(key_: string): Promise<string> { return ""; }

    public async updateTrialJob(_trialJobId: string, _form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        throw new Error('Not supported');
    }

    private async waitInitialized(): Promise<void> {
        if (this.initializePromise !== undefined) {
            await this.initializePromise;
            this.initializePromise = undefined;
        }
    }

    /**
     * upload local folder to nfs or azureStroage
     */
    private async uploadFolder(srcDirectory: string, destDirectory: string): Promise<string> {
        await this.waitInitialized();

        assert(this.config.storage.storage === 'azureStorage' || this.config.storage.storage === 'nfs');

        if (this.config.storage.storage === 'azureStorage') {
            if (this.azureStorageClient === undefined) {
                throw new Error('azureStorageClient is not initialized');
            }
            return await this.uploadFolderToAzureStorage(srcDirectory, destDirectory, this.config.deprecated.uploadRetryCount);
        } else if (this.config.storage.storage === 'nfs' || this.config.storage.storage === undefined) {
            await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}/${destDirectory}`);
            await cpp.exec(`cp -r ${srcDirectory}/* ${this.trialLocalNFSTempFolder}/${destDirectory}/.`);
            return `nfs://${this.config.storage.server}:${destDirectory}`;
        }
        return '';
    }

    private async prepareRunScript(trialLocalTempFolder: string, trialJobId: string, trialWorkingFolder: string,
                                   form: TrialJobApplicationForm): Promise<void> {
        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);
        const runScriptContent: string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        // Write worker file content run_worker.sh to local tmp folders
        if (this.config.worker !== undefined) {
            const workerRunScriptContent: string = await this.generateRunScript(
                'kubeflow',
                trialJobId,
                trialWorkingFolder,
                this.config.worker.command,
                form.sequenceId.toString(),
                'worker',
                this.config.worker.gpuNumber);
            await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_worker.sh'), workerRunScriptContent, { encoding: 'utf8' });
        }

        // Write parameter server file content run_ps.sh to local tmp folders
                                       
        if (this.config.parameterServer !== undefined) {
            const psName = this.config.operator === 'pytorch-operator' ? 'master' : 'ps';
            if (this.config.operator === 'tf-operator') {
               const psRunScriptContent: string = await this.generateRunScript(
                   'kubeflow',
                   trialJobId,
                   trialWorkingFolder,
                   this.config.parameterServer.command,
                   form.sequenceId.toString(),
                   psName,
                   this.config.parameterServer.gpuNumber);
               await fs.promises.writeFile(path.join(trialLocalTempFolder, `run_${psName}.sh`), psRunScriptContent, { encoding: 'utf8' });
           }
        }
        // Write file content ( parameter.cfg ) to local tmp folders
        if (form !== undefined) {
           await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(form.hyperParameters)),
                                       form.hyperParameters.value, { encoding: 'utf8' });
        }
    }

    private async prepareKubeflowConfig(trialJobId: string, trialWorkingFolder: string, kubeflowJobName: string): Promise<any> {
        await this.waitInitialized();

        const workerPodResources: any = {};
        if (this.config.worker !== undefined) {
            workerPodResources.requests = this.generatePodResource(
                toMegaBytes(this.config.worker.memorySize),
                this.config.worker.cpuNumber,
                this.config.worker.gpuNumber
            );
        }
        workerPodResources.limits = {...workerPodResources.requests};

        const nonWorkerResources: any = {};
        if (this.config.parameterServer !== undefined) {
            nonWorkerResources.requests = this.generatePodResource(
                toMegaBytes(this.config.parameterServer.memorySize),
                this.config.parameterServer.cpuNumber,
                this.config.parameterServer.gpuNumber
            );
            nonWorkerResources.limits = {...nonWorkerResources.requests};
        }

        // Generate kubeflow job resource config object
        const kubeflowJobConfig: any = await this.generateKubeflowJobConfig(trialJobId, trialWorkingFolder, kubeflowJobName, workerPodResources,
                                                                      nonWorkerResources);

        return Promise.resolve(kubeflowJobConfig);
    }

    /**
     * Generate kubeflow resource config file
     * @param trialJobId trial job id
     * @param trialWorkingFolder working folder
     * @param kubeflowJobName job name
     * @param workerPodResources worker pod template
     * @param nonWorkerPodResources non-worker pod template, like ps or master
     */
    private async generateKubeflowJobConfig(trialJobId: string, trialWorkingFolder: string, kubeflowJobName: string, workerPodResources: any,
                                            nonWorkerPodResources?: any): Promise<any> {
        const replicaSpecsObj: any = {};
        const replicaSpecsObjMap: Map<string, object> = new Map<string, object>();

        const privateRegistrySecretName = await this.createRegistrySecret(this.config.deprecated.privateRegistryAuthPath);
        const psName = this.config.operator === 'pytorch-operator' ? 'master' : 'ps';
        const upperPsName = this.config.operator === 'pytorch-operator' ? 'master' : 'ps';
        if (this.config.worker !== undefined) {
            replicaSpecsObj.Worker = this.generateReplicaConfig(
                trialWorkingFolder,
                this.config.worker.replicas,
                this.config.worker.dockerImage,
                'run_worker.sh',
                workerPodResources,
                privateRegistrySecretName
            );
        }
        if (this.config.parameterServer !== undefined) {
            replicaSpecsObj[upperPsName] = this.generateReplicaConfig(
                trialWorkingFolder,
                this.config.parameterServer.replicas,
                this.config.parameterServer.dockerImage,
                `run_${psName}.sh`,
                nonWorkerPodResources,
                privateRegistrySecretName
            );
        }

        if (this.config.operator === 'pytorch-operator') {
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, {pytorchReplicaSpecs: replicaSpecsObj});
        } else {
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, {tfReplicaSpecs: replicaSpecsObj});
        }

        return Promise.resolve({
            apiVersion: `kubeflow.org/${this.kubernetesCRDClient.apiVersion}`,
            kind: this.kubernetesCRDClient.jobKind,
            metadata: {
                name: kubeflowJobName,
                namespace: 'default',
                labels: {
                    app: this.NNI_KUBERNETES_TRIAL_LABEL,
                    expId: getExperimentId(),
                    trialId: trialJobId
                }
            },
            spec: replicaSpecsObjMap.get(this.kubernetesCRDClient.jobKind)
        });
    }

    /**
     * Generate tf-operator's tfjobs replica config section
     * @param trialWorkingFolder trial working folder
     * @param replicaNumber replica number
     * @param replicaImage image
     * @param runScriptFile script file name
     * @param podResources pod resource config section
     */
    private generateReplicaConfig(trialWorkingFolder: string, replicaNumber: number, replicaImage: string, runScriptFile: string,
                                  podResources: any, privateRegistrySecretName: string | undefined): any {
        // The config spec for volume field
        const volumeSpecMap: Map<string, object> = new Map<string, object>();
        if (this.config.storage.storage === 'azureStorage') {
            volumeSpecMap.set('nniVolumes', [
            {
                    name: 'nni-vol',
                    azureFile: {
                        secretName: `${this.azureStorageSecretName}`,
                        shareName: `${this.azureStorageShare}`,
                        readonly: false
                    }
            }]);
        } else {
            volumeSpecMap.set('nniVolumes', [
            {
                name: 'nni-vol',
                nfs: {
                    server: `${this.config.storage.server!}`,
                    path: `${this.config.storage.path!}`
                }
            }]);
        }
        // The config spec for container field
        const containersSpecMap: Map<string, object> = new Map<string, object>(); 
        containersSpecMap.set('containers', [
        {
                // Kubeflow tensorflow operator requires that containers' name must be tensorflow
                // TODO: change the name based on operator's type
                name: this.kubernetesCRDClient.containerName,
                image: replicaImage,
                args: ['sh', `${path.join(trialWorkingFolder, runScriptFile)}`],
                volumeMounts: [
                {
                    name: 'nni-vol',
                    mountPath: this.CONTAINER_MOUNT_PATH
                }],
                resources: podResources
            }
        ]);
        const spec: any = {
            containers: containersSpecMap.get('containers'),
            restartPolicy: 'ExitCode',
            volumes: volumeSpecMap.get('nniVolumes')
        }
        if (privateRegistrySecretName) {
            spec.imagePullSecrets = [
                {
                    name: privateRegistrySecretName
                }]
        }
        return {
            replicas: replicaNumber,
            template: {
                metadata: {
                    creationTimestamp: null
                },
                spec: spec
            }
        }
    }
}
export { KubeflowTrainingService };
