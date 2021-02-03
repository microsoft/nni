// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';

import { AzureStorage, KeyVaultConfig, KubernetesClusterConfig, KubernetesClusterConfigAzure, KubernetesClusterConfigNFS,
    KubernetesStorageKind, KubernetesTrialConfig, KubernetesTrialConfigTemplate, NFSConfig, StorageConfig, KubernetesClusterConfigPVC,
    PVCConfig,
} from '../kubernetesConfig';

export class FrameworkAttemptCompletionPolicy {
    public readonly minFailedTaskCount: number;
    public readonly minSucceededTaskCount: number;
    constructor(minFailedTaskCount: number, minSucceededTaskCount: number) {
        this.minFailedTaskCount = minFailedTaskCount;
        this.minSucceededTaskCount = minSucceededTaskCount;
    }
}

/**
 * Trial job configuration for FrameworkController
 */
export class FrameworkControllerTrialConfigTemplate extends KubernetesTrialConfigTemplate {
    public readonly frameworkAttemptCompletionPolicy: FrameworkAttemptCompletionPolicy;
    public readonly name: string;
    public readonly taskNum: number;
    constructor(taskNum: number, command: string, gpuNum: number,
                cpuNum: number, memoryMB: number, image: string,
                frameworkAttemptCompletionPolicy: FrameworkAttemptCompletionPolicy, privateRegistryFilePath?: string | undefined) {
        super(command, gpuNum, cpuNum, memoryMB, image, privateRegistryFilePath);
        this.frameworkAttemptCompletionPolicy = frameworkAttemptCompletionPolicy;
        this.name = name;
        this.taskNum = taskNum;
    }
}

export class FrameworkControllerTrialConfig extends KubernetesTrialConfig {
    public readonly taskRoles: FrameworkControllerTrialConfigTemplate[];
    public readonly codeDir: string;
    constructor(codeDir: string, taskRoles: FrameworkControllerTrialConfigTemplate[]) {
        super(codeDir);
        this.taskRoles = taskRoles;
        this.codeDir = codeDir;
    }
}

export class FrameworkControllerClusterConfig extends KubernetesClusterConfig {
    public readonly serviceAccountName: string;
    public readonly configPath?: string;
    constructor(apiVersion: string, serviceAccountName: string, configPath?: string) {
        super(apiVersion);
        this.serviceAccountName = serviceAccountName;
        this.configPath = configPath;
    }
}

export class FrameworkControllerClusterConfigPVC extends KubernetesClusterConfigPVC {
    public readonly serviceAccountName: string;
    public readonly configPath: string;
    constructor(serviceAccountName: string, apiVersion: string, pvc: PVCConfig, configPath: string, storage?: KubernetesStorageKind) {
        super(apiVersion, pvc, storage);
        this.serviceAccountName = serviceAccountName;
        this.configPath = configPath
    }

    public static getInstance(jsonObject: object): FrameworkControllerClusterConfigPVC {
        const kubernetesClusterConfigObjectPVC: FrameworkControllerClusterConfigPVC = <FrameworkControllerClusterConfigPVC>jsonObject;
        assert(kubernetesClusterConfigObjectPVC !== undefined);

        return new FrameworkControllerClusterConfigPVC(
            kubernetesClusterConfigObjectPVC.serviceAccountName,
            kubernetesClusterConfigObjectPVC.apiVersion,
            kubernetesClusterConfigObjectPVC.pvc,
            kubernetesClusterConfigObjectPVC.configPath,
            kubernetesClusterConfigObjectPVC.storage
        );
    }
}

export class FrameworkControllerClusterConfigNFS extends KubernetesClusterConfigNFS {
    public readonly serviceAccountName: string;
    constructor(
            serviceAccountName: string,
            apiVersion: string,
            nfs: NFSConfig,
            storage?: KubernetesStorageKind
        ) {
        super(apiVersion, nfs, storage);
        this.serviceAccountName = serviceAccountName;
    }

    public static getInstance(jsonObject: object): FrameworkControllerClusterConfigNFS {
        const kubeflowClusterConfigObjectNFS: FrameworkControllerClusterConfigNFS = <FrameworkControllerClusterConfigNFS>jsonObject;
        assert (kubeflowClusterConfigObjectNFS !== undefined);

        return new FrameworkControllerClusterConfigNFS(
            kubeflowClusterConfigObjectNFS.serviceAccountName,
            kubeflowClusterConfigObjectNFS.apiVersion,
            kubeflowClusterConfigObjectNFS.nfs,
            kubeflowClusterConfigObjectNFS.storage
        );
    }
}

export class FrameworkControllerClusterConfigAzure extends KubernetesClusterConfigAzure {
    public readonly serviceAccountName: string;

    constructor(
            serviceAccountName: string,
            apiVersion: string,
            keyVault: KeyVaultConfig,
            azureStorage: AzureStorage,
            storage?: KubernetesStorageKind
        ) {
        super(apiVersion, keyVault, azureStorage, storage);
        this.serviceAccountName = serviceAccountName;
    }

    public static getInstance(jsonObject: object): FrameworkControllerClusterConfigAzure {
        const kubeflowClusterConfigObjectAzure: FrameworkControllerClusterConfigAzure = <FrameworkControllerClusterConfigAzure>jsonObject;

        return new FrameworkControllerClusterConfigAzure(
            kubeflowClusterConfigObjectAzure.serviceAccountName,
            kubeflowClusterConfigObjectAzure.apiVersion,
            kubeflowClusterConfigObjectAzure.keyVault,
            kubeflowClusterConfigObjectAzure.azureStorage,
            kubeflowClusterConfigObjectAzure.storage
        );
    }
}

export class FrameworkControllerClusterConfigFactory {

    public static generateFrameworkControllerClusterConfig(jsonObject: object): FrameworkControllerClusterConfig {
         const storageConfig: StorageConfig = <StorageConfig>jsonObject;
         if (storageConfig === undefined) {
            throw new Error('Invalid json object as a StorageConfig instance');
        }
        if (storageConfig.storage !== undefined && storageConfig.storage === 'azureStorage') {
            return FrameworkControllerClusterConfigAzure.getInstance(jsonObject);
        } else if (storageConfig.storage === undefined || storageConfig.storage === 'nfs') {
            return FrameworkControllerClusterConfigNFS.getInstance(jsonObject);
        } else if ( storageConfig.storage === undefined || storageConfig.storage === 'pvc') {
            return FrameworkControllerClusterConfigPVC.getInstance(jsonObject);
        }
    throw new Error(`Invalid json object ${jsonObject}`);
  }
}

export type FrameworkControllerJobStatus =
  'AttemptRunning' | 'Completed' | 'AttemptCreationPending' | 'AttemptCreationRequested' | 'AttemptPreparing' | 'AttemptCompleted';

export type FrameworkControllerJobCompleteStatus = 'Succeeded' | 'Failed';
