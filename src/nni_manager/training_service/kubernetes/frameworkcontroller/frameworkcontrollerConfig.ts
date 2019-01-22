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

'use strict';
import * as assert from 'assert';

import { KubernetesTrialConfig, KubernetesTrialConfigTemplate, KubernetesClusterConfigAzure,
     KubernetesClusterConfigNFS, NFSConfig, KubernetesStorageKind, keyVaultConfig, AzureStorage, KubernetesClusterConfig,
    StorageConfig } from '../kubernetesConfig'

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
export class FrameworkControllerTrialConfigTemplate extends KubernetesTrialConfigTemplate{
    public readonly frameworkAttemptCompletionPolicy: FrameworkAttemptCompletionPolicy;
    public readonly name: string;
    public readonly taskNum: number;
    constructor(taskNum: number, command : string, gpuNum : number, 
        cpuNum: number, memoryMB: number, image: string, 
        frameworkAttemptCompletionPolicy: FrameworkAttemptCompletionPolicy) {
        super(command, gpuNum, cpuNum, memoryMB, image);
        this.frameworkAttemptCompletionPolicy = frameworkAttemptCompletionPolicy;
        this.name = name;
        this.taskNum = taskNum;
    }
}

export class FrameworkControllerTrialConfig extends KubernetesTrialConfig{
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
    constructor(apiVersion: string, serviceAccountName: string) {
        super(apiVersion);
        this.serviceAccountName = serviceAccountName;
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
        let kubeflowClusterConfigObjectNFS = <FrameworkControllerClusterConfigNFS>jsonObject;
        assert (kubeflowClusterConfigObjectNFS !== undefined)
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
            keyVault: keyVaultConfig, 
            azureStorage: AzureStorage, 
            storage?: KubernetesStorageKind
        ) {
        super(apiVersion, keyVault, azureStorage,storage);
        this.serviceAccountName = serviceAccountName;
    }

    public static getInstance(jsonObject: object): FrameworkControllerClusterConfigAzure {
        let kubeflowClusterConfigObjectAzure = <FrameworkControllerClusterConfigAzure>jsonObject;
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
         let storageConfig = <StorageConfig>jsonObject;
         if(!storageConfig) {
            throw new Error("Invalid json object as a StorageConfig instance");
        }
         if(storageConfig.storage && storageConfig.storage === 'azureStorage') {
            return FrameworkControllerClusterConfigAzure.getInstance(jsonObject);
         } else if (storageConfig.storage === undefined || storageConfig.storage === 'nfs') {
            return FrameworkControllerClusterConfigNFS.getInstance(jsonObject);
         }
         throw new Error(`Invalid json object ${jsonObject}`);
    }
}

export type FrameworkControllerJobStatus = 'AttemptRunning' | 'Completed' | 'AttemptCreationPending' | 'AttemptCreationRequested' | 'AttemptPreparing' | 'AttemptCompleted';

export type FrameworkControllerJobCompleteStatus = 'Succeeded' | 'Failed';