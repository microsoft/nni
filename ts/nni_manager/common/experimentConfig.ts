// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert';

import { KubeflowOperator, OperatorApiVersion } from '../training_service/kubernetes/kubeflow/kubeflowConfig'
import { KubernetesStorageKind } from '../training_service/kubernetes/kubernetesConfig';

export interface TrainingServiceConfig {
    platform: string;
}

/* Local */

export interface LocalConfig extends TrainingServiceConfig {
    platform: 'local';
    reuseMode: boolean;
    useActiveGpu?: boolean;
    maxTrialNumberPerGpu: number;
    gpuIndices?: number[];
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
    reuseMode: boolean;
    machineList: RemoteMachineConfig[];
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
    localStorageMountPoint: string;
    containerStorageMountPoint: string;
    reuseMode: boolean;
    openpaiConfig?: object;
    virtualCluster?: string;
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
    nasDataSourceId: string;
    accessKeyId: string;
    accessKeySecret: string;
    localStorageMountPoint: string;
    containerStorageMountPoint: string;
}
/* Kubeflow */

// FIXME: merge with shared storage config
export interface KubeflowStorageConfig {
    storageType: string;
    maxTrialNumberPerGpu?: number;
    server?: string;
    path?: string;
    azureAccount?: string;
    azureShare?: string;
    keyVaultName?: string;
    keyVaultKey?: string;
}

export interface KubeflowRoleConfig {
    replicas: number;
    codeDirectory: string;
    command: string;
    gpuNumber: number;
    cpuNumber: number;
    memorySize: number;
    dockerImage: string;
    privateRegistryAuthPath?: string;
}

export interface KubeflowConfig extends TrainingServiceConfig {
    platform: 'kubeflow';
    ps?: KubeflowRoleConfig;
    master?: KubeflowRoleConfig;
    worker?: KubeflowRoleConfig;
    maxTrialNumberPerGpu: number;
    operator: KubeflowOperator;
    apiVersion: OperatorApiVersion;
    storage: KubeflowStorageConfig;
    reuseMode: boolean;
}

/* FrameworkController */

type FrameworkControllerStorageConfig = KubeflowStorageConfig;

export interface FrameworkControllerRoleConfig {
    name: string;
    dockerImage: string;
    taskNumber: number;
    command: string;
    gpuNumber: number;
    cpuNumber: number;
    memorySize: string;
    attemptCompletionMinFailedTasks: number;
    attemptCompletionMinSucceededTasks: number;
}

export interface FrameworkControllerConfig extends TrainingServiceConfig {
    platform: 'frameworkcontroller';
    serviceAccountName: string;
    storage: FrameworkControllerStorageConfig;
    taskRoles: FrameworkControllerRoleConfig[];
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
    searchSpace: any;
    trialCommand: string;
    trialCodeDirectory: string;
    trialConcurrency: number;
    trialGpuNumber?: number;
    maxExperimentDuration?: string;
    maxTrialDuration?: string;
    maxTrialNumber?: number;
    nniManagerIp?: string;
    //useAnnotation: boolean;  // dealed inside nnictl
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

export function toSeconds(time: string): number {
    for (const [unit, factor] of Object.entries(timeUnits)) {
        if (time.toLowerCase().endsWith(unit)) {
            const digits = time.slice(0, -1);
            return Number(digits) * factor;
        }
    }
    throw new Error(`Bad time string "${time}"`);
}

const sizeUnits = { tb: 1024 * 1024, gb: 1024, mb: 1, kb: 1 / 1024 };

export function toMegaBytes(size: string): number {
    for (const [unit, factor] of Object.entries(sizeUnits)) {
        if (size.toLowerCase().endsWith(unit)) {
            const digits = size.slice(0, -2);
            return Math.floor(Number(digits) * factor);
        }
    }
    throw new Error(`Bad size string "${size}"`);
}

export function toCudaVisibleDevices(gpuIndices?: number[]): string {
    return gpuIndices === undefined ? '' : gpuIndices.join(',');
}

export function flattenConfig<T>(config: ExperimentConfig, platform: string): T {
    const flattened = { };
    Object.assign(flattened, config);
    if (Array.isArray(config.trainingService)) {
        for (const trainingService of config.trainingService) {
            if (trainingService.platform === platform) {
                Object.assign(flattened, trainingService);
            }
        }
    } else {
        assert(config.trainingService.platform === platform);
        Object.assign(flattened, config.trainingService);
    }
    return <T>flattened;
}
