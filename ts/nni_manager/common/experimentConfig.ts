// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { KubeflowOperator, OperatorApiVersion } from '../training_service/kubernetes/kubeflow/kubeflowConfig'

export interface TrainingServiceConfig {
    platform: string;
    trialCommand: string;
    trialCodeDirectory: string;
    trialGpuNumber?: number;
    nniManagerIp?: string;

    // FIXME
    // "debug" is only used by openpai to decide whether to check remote nni version
    // it should be better to check when local nni version is not "dev"
    // it should be even better to check version before launching the experiment and let user to confirm
    // log level is currently handled by global logging module and has nothing to do with this
    debug?: boolean;
}

/* Local */

export interface LocalConfig extends TrainingServiceConfig {
    platform: 'local';
    useActiveGpu?: boolean;
    maxTrialNumberPerGpu: number;
    gpuIndices?: number[];
    reuseMode: boolean;
}

/* Remote */

export interface RemoteMachineConfig {
    host: string;
    port: number;
    user: string;
    password?: string;
    sshKeyFile: string;
    sshPassphrase?: string;
    useActiveGpu: boolean;
    maxTrialNumberPerGpu: number;
    gpuIndices?: number[];
    pythonPath?: string;
}

export interface RemoteConfig extends TrainingServiceConfig {
    platform: 'remote';
    machineList: RemoteMachineConfig[];
    reuseMode: boolean;
    //logCollection: 'on_error' | 'always' | 'never'
}

/* OpenPAI */

export interface OpenpaiConfig extends TrainingServiceConfig {
    platform: 'openpai';
    host: string;
    username: string;
    token: string;
    trialCpuNumber: number;
    trialMemorySize: string;
    storageConfigName: string;
    dockerImage: string;
    virtualCluster?: string;
    localStorageMountPoint: string;
    containerStorageMountPoint: string;
    reuseMode: boolean;
    openpaiConfig?: object;
}

/* AML */

export interface AmlConfig extends TrainingServiceConfig {
    platform: 'aml';
    subscriptionId: string;
    resourceGroup: string;
    workspaceName: string;
    computeTarget: string;
    dockerImage: string;
    maxTrialNumberPerGpu: number;
}


/*  Alibaba PAI DLC  */
export interface DlcConfig extends TrainingServiceConfig {
    platfrom: 'dlc';
    type: string;
    image: string;
    jobType: string;
    podCount: number;
    ecsSpec: string;
    region: string;
    workspaceId: string;
    nasDataSourceId: string;
    ossDataSourceId?: string;
    accessKeyId: string;
    accessKeySecret: string;
    localStorageMountPoint: string;
    containerStorageMountPoint: string;
}
/* Kubeflow */

export interface KubernetesStorageConfig {
    storageType: string;
    server?: string;
    path?: string;
    azureAccount?: string;
    azureShare?: string;
    keyVaultName?: string;
    keyVaultKey?: string;
}

export interface KubeflowRoleConfig {
    replicas: number;
    command: string;
    gpuNumber: number;
    cpuNumber: number;
    memorySize: string | number;
    dockerImage: string;
    codeDirectory: string;
    privateRegistryAuthPath?: string;
}

export interface KubeflowConfig extends TrainingServiceConfig {
    platform: 'kubeflow';
    operator: KubeflowOperator;
    apiVersion: OperatorApiVersion;
    storage: KubernetesStorageConfig;
    worker?: KubeflowRoleConfig;
    ps?: KubeflowRoleConfig;
    master?: KubeflowRoleConfig;
    reuseMode: boolean;
    maxTrialNumberPerGpu?: number;
    namespace?: string;
}

export interface FrameworkControllerTaskRoleConfig {
    name: string;
    dockerImage: string;
    taskNumber: number;
    command: string;
    gpuNumber: number;
    cpuNumber: number;
    memorySize: string | number;
    frameworkAttemptCompletionPolicy: {
        minFailedTaskCount: number;
        minSucceedTaskCount: number;
    };
    privateRegistryAuthPath?: string;
}

export interface FrameworkControllerConfig extends TrainingServiceConfig {
    platform: 'frameworkcontroller';
    storage: KubernetesStorageConfig;
    serviceAccountName: string;
    taskRoles: FrameworkControllerTaskRoleConfig[];
    reuseMode: boolean;
    maxTrialNumberPerGpu?: number;
    namespace?: string;
    apiVersion?: string;
}

/* shared storage */

export interface SharedStorageConfig {
    storageType: string;
    localMountPoint: string;
    remoteMountPoint: string;
    localMounted: string;
}

export interface NfsConfig extends SharedStorageConfig {
    storageType: 'NFS';
    nfsServer: string;
    exportedDirectory: string;
}

export interface AzureBlobConfig extends SharedStorageConfig {
    storageAccountName: string;
    storageAccountKey?: string;
    containerName: string;
}

/* common */

export interface AlgorithmConfig {
    name?: string;
    className?: string;
    codeDirectory?: string;
    classArgs?: object;
}

export interface ExperimentConfig {
    experimentName?: string;
    // searchSpaceFile  (handled in python part)
    searchSpace: any;
    trialCommand: string;
    trialCodeDirectory: string;
    trialConcurrency: number;
    trialGpuNumber?: number;
    maxExperimentDuration?: string | number;
    maxTrialNumber?: number;
    maxTrialDuration?: string | number;
    nniManagerIp?: string;
    // useAnnotation  (handled in python part)
    debug: boolean;
    logLevel?: string;
    experimentWorkingDirectory?: string;
    tunerGpuIndices?: number[];
    tuner?: AlgorithmConfig;
    assessor?: AlgorithmConfig;
    advisor?: AlgorithmConfig;
    trainingService: TrainingServiceConfig | TrainingServiceConfig[];
    sharedStorage?: SharedStorageConfig;
    deprecated?: any;  // configs that are not yet natively supported by v2 (workaround)
}

/* util functions */

const timeUnits = { d: 24 * 3600, h: 3600, m: 60, s: 1 };
const sizeUnits = { tb: 1024 ** 4, gb: 1024 ** 3, mb: 1024 ** 2, kb: 1024, b: 1 };

function toUnit(value: string | number, targetUnit: string, allUnits: any): number {
    if (typeof value === 'number') {
        return value;
    }
    value = value.toLowerCase();
    for (const [unit, factor] of Object.entries(allUnits)) {
        if (value.endsWith(unit)) {
            const digits = value.slice(0, -unit.length);
            const num = Number(digits) * (factor as number);
            return Math.ceil(num / allUnits[targetUnit]);
        }
    }
    throw new Error(`Bad unit in "${value}"`);
}

export function toSeconds(time: string | number): number {
    return toUnit(time, 's', timeUnits);
}

export function toMegaBytes(size: string | number): number {
    return toUnit(size, 'mb', sizeUnits);
}

export function toCudaVisibleDevices(gpuIndices?: number[]): string {
        return gpuIndices === undefined ? '' : gpuIndices.join(',');
}
