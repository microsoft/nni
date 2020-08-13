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
    NNIManagerIpConfig, TrialJobApplicationForm, TrialJobDetail, TrialJobStatus
} from '../../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, uniqueString } from '../../../common/utils';
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

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class KubeflowTrainingService extends KubernetesTrainingService implements KubernetesTrainingService {
    private kubeflowClusterConfig?: KubeflowClusterConfig;
    private kubeflowTrialConfig?: KubeflowTrialConfig;
    private readonly kubeflowJobInfoCollector: KubeflowJobInfoCollector;

    constructor() {
        super();
        this.kubeflowJobInfoCollector = new KubeflowJobInfoCollector(this.trialJobsMap);
        this.experimentId = getExperimentId();
        this.log.info('Construct Kubeflow training service.');
    }

    public async run(): Promise<void> {
        this.log.info('Run Kubeflow training service.');
        this.kubernetesJobRestServer = component.get(KubeflowJobRestServer);
        if (this.kubernetesJobRestServer === undefined) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        await this.kubernetesJobRestServer.start();
        this.kubernetesJobRestServer.setEnableVersionCheck = this.versionCheck;
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
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Kubeflow job operator client is undefined');
        }

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

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.KUBEFLOW_CLUSTER_CONFIG: {
                const kubeflowClusterJsonObject: object = JSON.parse(value);
                this.kubeflowClusterConfig = KubeflowClusterConfigFactory.generateKubeflowClusterConfig(kubeflowClusterJsonObject);
                if (this.kubeflowClusterConfig.storageType === 'azureStorage') {
                    const azureKubeflowClusterConfig: KubeflowClusterConfigAzure = <KubeflowClusterConfigAzure>this.kubeflowClusterConfig;
                    this.azureStorageAccountName = azureKubeflowClusterConfig.azureStorage.accountName;
                    this.azureStorageShare = azureKubeflowClusterConfig.azureStorage.azureShare;
                    await this.createAzureStorage(
                        azureKubeflowClusterConfig.keyVault.vaultName,
                        azureKubeflowClusterConfig.keyVault.name
                    );
                } else if (this.kubeflowClusterConfig.storageType === 'nfs') {
                    const nfsKubeflowClusterConfig: KubeflowClusterConfigNFS = <KubeflowClusterConfigNFS>this.kubeflowClusterConfig;
                    await this.createNFSStorage(
                        nfsKubeflowClusterConfig.nfs.server,
                        nfsKubeflowClusterConfig.nfs.path
                    );
                }
                this.kubernetesCRDClient = KubeflowOperatorClientFactory.createClient(
                    this.kubeflowClusterConfig.operator, this.kubeflowClusterConfig.apiVersion);
                break;
            }
            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                if (this.kubeflowClusterConfig === undefined) {
                    this.log.error('kubeflow cluster config is not initialized');

                    return Promise.reject(new Error('kubeflow cluster config is not initialized'));
                }

                assert(this.kubeflowClusterConfig !== undefined);
                const kubeflowTrialJsonObjsect: object = JSON.parse(value);
                this.kubeflowTrialConfig = KubeflowTrialConfigFactory.generateKubeflowTrialConfig(
                    kubeflowTrialJsonObjsect,
                    this.kubeflowClusterConfig.operator
                );

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.kubeflowTrialConfig.codeDir);
                    //upload codeDir to storage
                    this.copyExpCodeDirPromise = this.uploadFolder(this.kubeflowTrialConfig.codeDir, `nni/${getExperimentId()}/nni-code`);
                } catch (error) {
                    this.log.error(error);

                    return Promise.reject(new Error(error));
                }
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

    /**
     * upload local folder to nfs or azureStroage
     */
    private async uploadFolder(srcDirectory: string, destDirectory: string): Promise<string> {
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if (this.kubeflowTrialConfig === undefined) {
            throw new Error('Kubeflow Trial config is not initialized');
        }

        assert(this.kubeflowClusterConfig.storage === undefined
            || this.kubeflowClusterConfig.storage === 'azureStorage'
            || this.kubeflowClusterConfig.storage === 'nfs');

        if (this.kubeflowClusterConfig.storage === 'azureStorage') {
            if (this.azureStorageClient === undefined) {
                throw new Error('azureStorageClient is not initialized');
            }
            const azureKubeflowClusterConfig: KubeflowClusterConfigAzure = <KubeflowClusterConfigAzure>this.kubeflowClusterConfig;
            return await this.uploadFolderToAzureStorage(srcDirectory, destDirectory, azureKubeflowClusterConfig.uploadRetryCount);
        } else if (this.kubeflowClusterConfig.storage === 'nfs' || this.kubeflowClusterConfig.storage === undefined) {
            await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}/${destDirectory}`);
            await cpp.exec(`cp -r ${srcDirectory}/* ${this.trialLocalNFSTempFolder}/${destDirectory}/.`);
            const nfsKubeflowClusterConfig: KubeflowClusterConfigNFS = <KubeflowClusterConfigNFS>this.kubeflowClusterConfig;
            const nfsConfig: NFSConfig = nfsKubeflowClusterConfig.nfs;
            return `nfs://${nfsConfig.server}:${destDirectory}`;
        }
        return '';
    }

    private async prepareRunScript(trialLocalTempFolder: string, trialJobId: string, trialWorkingFolder: string,
                                   form: TrialJobApplicationForm): Promise<void> {
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        // initialize kubeflow trial config to specific type
        let kubeflowTrialConfig: any;
        if (this.kubeflowClusterConfig.operator === 'tf-operator') {
            kubeflowTrialConfig = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
        } else if (this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            kubeflowTrialConfig = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
        } else {
            throw Error(`operator ${this.kubeflowClusterConfig.operator} is invalid`);
        }

        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);
        const runScriptContent: string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        // Write worker file content run_worker.sh to local tmp folders
        if (kubeflowTrialConfig.worker !== undefined) {
           const workerRunScriptContent: string = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder,
                                                                               kubeflowTrialConfig.worker.command,
                                                                               form.sequenceId.toString(), 'worker',
                                                                               kubeflowTrialConfig.worker.gpuNum);
           await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_worker.sh'), workerRunScriptContent, { encoding: 'utf8' });
        }
        // Write parameter server file content run_ps.sh to local tmp folders
        if (this.kubeflowClusterConfig.operator === 'tf-operator') {
           const tensorflowTrialConfig: KubeflowTrialConfigTensorflow = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
           if (tensorflowTrialConfig.ps !== undefined) {
               const psRunScriptContent: string = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder,
                                                                               tensorflowTrialConfig.ps.command,
                                                                               form.sequenceId.toString(),
                                                                               'ps', tensorflowTrialConfig.ps.gpuNum);
               await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_ps.sh'), psRunScriptContent, { encoding: 'utf8' });
           }
        } else if (this.kubeflowClusterConfig.operator === 'pytorch-operator') {
           const pytorchTrialConfig: KubeflowTrialConfigPytorch = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
           if (pytorchTrialConfig.master !== undefined) {
               const masterRunScriptContent: string = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder,
                                                                                   pytorchTrialConfig.master.command,
                                                                                   form.sequenceId.toString(), 'master',
                                                                                   pytorchTrialConfig.master.gpuNum);
               await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_master.sh'), masterRunScriptContent, { encoding: 'utf8' });
           }
        }
        // Write file content ( parameter.cfg ) to local tmp folders
        if (form !== undefined) {
           await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(form.hyperParameters)),
                                       form.hyperParameters.value, { encoding: 'utf8' });
        }
    }

    private async prepareKubeflowConfig(trialJobId: string, trialWorkingFolder: string, kubeflowJobName: string): Promise<any> {
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if (this.kubeflowTrialConfig === undefined) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        // initialize kubeflow trial config to specific type
        let kubeflowTrialConfig: any;
        if (this.kubeflowClusterConfig.operator === 'tf-operator') {
            kubeflowTrialConfig = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
        } else if (this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            kubeflowTrialConfig = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
        } else {
            throw Error(`operator ${this.kubeflowClusterConfig.operator} is invalid`);
        }

        const workerPodResources: any = {};
        if (kubeflowTrialConfig.worker !== undefined) {
            workerPodResources.requests = this.generatePodResource(kubeflowTrialConfig.worker.memoryMB, kubeflowTrialConfig.worker.cpuNum,
                                                                   kubeflowTrialConfig.worker.gpuNum);
        }
        workerPodResources.limits = {...workerPodResources.requests};

        const nonWorkerResources: any = {};
        if (this.kubeflowClusterConfig.operator === 'tf-operator') {
            const tensorflowTrialConfig: KubeflowTrialConfigTensorflow = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
            if (tensorflowTrialConfig.ps !== undefined) {
                nonWorkerResources.requests = this.generatePodResource(tensorflowTrialConfig.ps.memoryMB, tensorflowTrialConfig.ps.cpuNum,
                                                                       tensorflowTrialConfig.ps.gpuNum);
                nonWorkerResources.limits = {...nonWorkerResources.requests};
            }
        } else if (this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            const pyTorchTrialConfig: KubeflowTrialConfigPytorch = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
            nonWorkerResources.requests = this.generatePodResource(pyTorchTrialConfig.master.memoryMB, pyTorchTrialConfig.master.cpuNum,
                                                                   pyTorchTrialConfig.master.gpuNum);
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
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if (this.kubeflowTrialConfig === undefined) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Kubeflow operator client is not initialized');
        }

        const replicaSpecsObj: any = {};
        const replicaSpecsObjMap: Map<string, object> = new Map<string, object>();
        if (this.kubeflowTrialConfig.operatorType === 'tf-operator') {
            const tensorflowTrialConfig: KubeflowTrialConfigTensorflow = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
            const privateRegistrySecretName = await this.createRegistrySecret(tensorflowTrialConfig.worker.privateRegistryAuthPath);
            replicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, tensorflowTrialConfig.worker.replicas,
                                                                tensorflowTrialConfig.worker.image, 'run_worker.sh', workerPodResources, privateRegistrySecretName);
            if (tensorflowTrialConfig.ps !== undefined) {
                const privateRegistrySecretName: string | undefined = await this.createRegistrySecret(tensorflowTrialConfig.ps.privateRegistryAuthPath);
                replicaSpecsObj.Ps = this.generateReplicaConfig(trialWorkingFolder, tensorflowTrialConfig.ps.replicas,
                                                                tensorflowTrialConfig.ps.image, 'run_ps.sh', nonWorkerPodResources, privateRegistrySecretName);
            }
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, {tfReplicaSpecs: replicaSpecsObj});
        } else if (this.kubeflowTrialConfig.operatorType === 'pytorch-operator') {
            const pytorchTrialConfig: KubeflowTrialConfigPytorch = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
            if (pytorchTrialConfig.worker !== undefined) {
                const privateRegistrySecretName: string | undefined = await this.createRegistrySecret(pytorchTrialConfig.worker.privateRegistryAuthPath);
                replicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, pytorchTrialConfig.worker.replicas,
                                                                    pytorchTrialConfig.worker.image, 'run_worker.sh', workerPodResources, privateRegistrySecretName);
            }
            const privateRegistrySecretName: string | undefined = await this.createRegistrySecret(pytorchTrialConfig.master.privateRegistryAuthPath);
            replicaSpecsObj.Master = this.generateReplicaConfig(trialWorkingFolder, pytorchTrialConfig.master.replicas,
                                                                pytorchTrialConfig.master.image, 'run_master.sh', nonWorkerPodResources, privateRegistrySecretName);

            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, {pytorchReplicaSpecs: replicaSpecsObj});
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
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if (this.kubeflowTrialConfig === undefined) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Kubeflow operator client is not initialized');
        }
        // The config spec for volume field
        const volumeSpecMap: Map<string, object> = new Map<string, object>();
        if (this.kubeflowClusterConfig.storageType === 'azureStorage') {
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
            const nfsKubeflowClusterConfig: KubeflowClusterConfigNFS = <KubeflowClusterConfigNFS> this.kubeflowClusterConfig;
            volumeSpecMap.set('nniVolumes', [
            {
                name: 'nni-vol',
                nfs: {
                    server: `${nfsKubeflowClusterConfig.nfs.server}`,
                    path: `${nfsKubeflowClusterConfig.nfs.path}`
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
