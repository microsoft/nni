// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

export interface TrainingServiceConfig {
    platform: string;
};

/* Local */

export interface LocalConfig extends TrainingServiceConfig {
    platform: 'local';
    useActiveGpu?: boolean;
    maxTrialNumberPerGpu: number;
    gpuIndices?: number[];
};

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
};

export interface RemoteConfig extends TrainingServiceConfig {
    platform: 'remote';
    reuseMode: boolean;
    machineList: RemoteMachineConfig[];
};

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
    openpaiConfig?: Object;
};

/* AML */

export interface AmlConfig extends TrainingServiceConfig {
    platform: 'aml';
    subscriptionId: string;
    resourceGroup: string;
    workspaceName: string;
    computeTarget: string;
    dockerImage: string;
};

/* Kubeflow */

export interface KubeflowStorageConfig {
    storage: string;
    server?: string;
    path?: string;
    azureAccount?: string;
    azureShare?: string;
    keyVault?: string;
    keyVaultSecret?: string;
};

export interface KubeflowRoleConfig {
    replicas: number;
    command: string;
    gpuNumber: number;
    cpuNumber: number;
    memorySize: string;
    dockerImage: string;
};

export interface KubeflowConfig extends TrainingServiceConfig {
    platform: 'kubeflow';
    operator: string;
    apiVersion: string;
    storage: KubeflowStorageConfig;
    worker: KubeflowRoleConfig;
    parameterServer?: KubeflowRoleConfig;
};

/* FrameworkController */

export interface FrameworkControllerStorageConfig extends KubeflowStorageConfig { };

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
};

export interface FrameworkControllerConfig extends TrainingServiceConfig {
    platform: 'frameworkcontroller';
    serviceAccountName: string;
    storage: FrameworkControllerStorageConfig;
    taskRoles: FrameworkControllerRoleConfig[];
};

/* common */

export interface AlgorithmConfig {
    name?: string;
    className?: string;
    codeDirectory?: string;
    classArgs?: Object;
};

export interface ExperimentConfig {
    experimentName?: string;
    searchSpace: any;
    trialCommand: string;
    trialCodeDirectory: string;
    trialConcurrency: number;
    trialGpuNumber?: number;
    maxExperimentDuration?: string;
    maxTrialNumber?: number;
    nniManagerIp?: string;
    //useAnnotation: boolean;
    debug: boolean;
    logLevel?: string;
    experimentWorkingDirectory?: string;
    tunerGpuIndices?: number[];
    tuner?: AlgorithmConfig;
    assessor?: AlgorithmConfig;
    advisor?: AlgorithmConfig;
    trainingService: TrainingServiceConfig;
};

/* util functions */

const time_units = { 'd': 24 * 3600, 'h': 3600, 'm': 60, 's': 1 };

export function toSeconds(time: string): number {
    for (let [unit, factor] of Object.entries(time_units)) {
        if (time.endsWith(unit)) {
            const digits = time.slice(0, -1);
            return Number(digits) * factor;
        }
    }
    throw new Error(`Bad time string "${time}"`);
};

export function toCudaVisibleDevices(gpuIndices?: number[]): string {
    return gpuIndices === undefined ? '' : gpuIndices.join(',');
}

export function flattenConfig<T>(config: ExperimentConfig): T {
    const flattened = { };
    Object.assign(flattened, config);
    Object.assign(flattened, config.trainingService);
    return <T>flattened;
}
