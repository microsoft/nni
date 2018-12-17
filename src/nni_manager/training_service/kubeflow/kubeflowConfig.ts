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

import { KubernetesClusterConfig, KubernetesStorageKind, NFSConfig, AzureStorage, keyVaultConfig,
        KubernetesTrialConfig, KubernetesTrialConfigTemplate } from '../kubernetes/kubernetesConfig'

/** operator types that kubeflow supported */
export type KubeflowOperator = 'tf-operator' | 'pytorch-operator' ;
export type KubeflowStorageKind = 'nfs' | 'azureStorage';
export type DistTrainRole = 'worker' | 'ps' | 'master';
export type OperatorApiVersion = 'v1alpha2' | 'v1beta1';
export type KubernetesJobType = 'Created' | 'Running' | 'Failed' | 'Succeeded';

/**
 * Kuberflow cluster configuration
 * 
 */
export class KubeflowClusterConfigBase extends KubernetesClusterConfig {
    /** Name of Kubeflow operator, like tf-operator */
    public readonly operator: KubeflowOperator;
    public readonly apiVersion: OperatorApiVersion;
    public readonly storage?: KubeflowStorageKind;    
    
    /**
     * Constructor
     * @param userName User name of Kubeflow Cluster
     * @param passWord password of Kubeflow Cluster
     * @param host Host IP of Kubeflow Cluster
     */
    constructor(operator: KubeflowOperator, apiVersion: OperatorApiVersion, storage?: KubeflowStorageKind) {
        super(storage)
        this.operator = operator;
        this.apiVersion = apiVersion;
        this.storage = storage;
    }
}

export class KubeflowClusterConfigNFS extends KubeflowClusterConfigBase{
    public readonly nfs: NFSConfig;
    
    constructor(operator: KubeflowOperator, 
            apiVersion: OperatorApiVersion, 
            nfs: NFSConfig, storage?: KubeflowStorageKind) {
        super(operator, apiVersion, storage);
        this.nfs = nfs;
    }
}

export class KubeflowClusterConfigAzure extends KubeflowClusterConfigBase{
    public readonly keyVault: keyVaultConfig;
    public readonly azureStorage: AzureStorage;
    
    constructor(operator: KubeflowOperator, 
            apiVersion: OperatorApiVersion, 
            keyVault: keyVaultConfig, 
            azureStorage: AzureStorage, 
            storage?: KubeflowStorageKind) {
        super(operator, apiVersion, storage);
        this.keyVault = keyVault;
        this.azureStorage = azureStorage;
    }
}

export class KubeflowTrialConfigTensorflow extends KubernetesTrialConfig{
    public readonly ps?: KubernetesTrialConfigTemplate;
    public readonly worker: KubernetesTrialConfigTemplate;

    constructor(codeDir: string, worker: KubernetesTrialConfigTemplate,  ps?: KubernetesTrialConfigTemplate) {
        super(codeDir);
        this.ps = ps;
        this.worker = worker;
    }
}

export class KubeflowTrialConfigPytorch extends KubernetesTrialConfig{
    public readonly master: KubernetesTrialConfigTemplate;
    public readonly worker?: KubernetesTrialConfigTemplate;

    constructor(codeDir: string, master: KubernetesTrialConfigTemplate, worker?: KubernetesTrialConfigTemplate) {
        super(codeDir);
        this.master = master;
        this.worker = worker;
    }
}

