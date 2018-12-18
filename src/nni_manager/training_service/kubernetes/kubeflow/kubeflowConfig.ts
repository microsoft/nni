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
        KubernetesTrialConfig, KubernetesTrialConfigTemplate } from '../kubernetesConfig'
import { MethodNotImplementedError } from '../../../common/errors';

/** operator types that kubeflow supported */
export type KubeflowOperator = 'tf-operator' | 'pytorch-operator' ;
export type DistTrainRole = 'worker' | 'ps' | 'master';
export type OperatorApiVersion = 'v1alpha2' | 'v1beta1';
export type KubeflowJobType = 'Created' | 'Running' | 'Failed' | 'Succeeded';

/**
 * Kuberflow cluster configuration
 * 
 */
export class KubeflowClusterConfigBase extends KubernetesClusterConfig {
    /** Name of Kubeflow operator, like tf-operator */
    public readonly operator: KubeflowOperator;
    public readonly apiVersion: OperatorApiVersion;
    public readonly storage?: KubernetesStorageKind;    
    
    /**
     * Constructor
     * @param userName User name of Kubeflow Cluster
     * @param passWord password of Kubeflow Cluster
     * @param host Host IP of Kubeflow Cluster
     */
    constructor(operator: KubeflowOperator, apiVersion: OperatorApiVersion, storage?: KubernetesStorageKind) {
        super(storage)
        this.operator = operator;
        this.apiVersion = apiVersion;
        this.storage = storage;
    }
}

export class KubeflowClusterConfigNFS extends KubeflowClusterConfigBase{
    public readonly nfs: NFSConfig;
    
    constructor(
            operator: KubeflowOperator, 
            apiVersion: OperatorApiVersion, 
            nfs: NFSConfig,
            storage?: KubernetesStorageKind
        ) {
        super(operator, apiVersion, storage);
        this.nfs = nfs;
    }

    public get storageType(): KubernetesStorageKind{
        return 'nfs';
    }
}

export class KubeflowClusterConfigAzure extends KubeflowClusterConfigBase{
    public readonly keyVault: keyVaultConfig;
    public readonly azureStorage: AzureStorage;
    
    constructor(
            operator: KubeflowOperator, 
            apiVersion: OperatorApiVersion, 
            keyVault: keyVaultConfig, 
            azureStorage: AzureStorage, 
            storage?: KubernetesStorageKind
        ) {
        super(operator, apiVersion, storage);
        this.keyVault = keyVault;
        this.azureStorage = azureStorage;
    }

    public get storageType(): KubernetesStorageKind{
        return 'azureStorage';
    }
}

export class KubeflowClusterConfigFactory {

    public static generateKubeflowClusterConfig(jsonObject: object): KubeflowClusterConfigBase {
         let kubeflowClusterConfigObject = <KubeflowClusterConfigBase>jsonObject;
         if(kubeflowClusterConfigObject.storage && kubeflowClusterConfigObject.storage === 'azureStorage') {
            let kubeflowClusterConfigAzureObject = <KubeflowClusterConfigAzure>jsonObject;
            return new KubeflowClusterConfigAzure(
                kubeflowClusterConfigAzureObject.operator, 
                kubeflowClusterConfigAzureObject.apiVersion,
                kubeflowClusterConfigAzureObject.keyVault,
                kubeflowClusterConfigAzureObject.azureStorage,
                kubeflowClusterConfigAzureObject.storage);
         } else if (kubeflowClusterConfigObject.storage === undefined || kubeflowClusterConfigObject.storage === 'nfs') {
            let kubeflowClusterConfigNFSObject = <KubeflowClusterConfigNFS>jsonObject;
            return new KubeflowClusterConfigNFS(
                kubeflowClusterConfigNFSObject.operator,
                kubeflowClusterConfigNFSObject.apiVersion,
                kubeflowClusterConfigNFSObject.nfs,
                kubeflowClusterConfigNFSObject.storage
                );
         }
         throw new Error(`Invalid json object ${jsonObject}`);
    }
}

export class KubeflowTrialConfig extends KubernetesTrialConfig {
    constructor(codeDir: string) {
        super(codeDir);
    }

    public get operatorType(): KubeflowOperator {
        throw new MethodNotImplementedError();
    }
}

export class KubeflowTrialConfigTensorflow extends KubeflowTrialConfig {
    public readonly ps?: KubernetesTrialConfigTemplate;
    public readonly worker: KubernetesTrialConfigTemplate;

    constructor(codeDir: string, worker: KubernetesTrialConfigTemplate,  ps?: KubernetesTrialConfigTemplate) {
        super(codeDir);
        this.ps = ps;
        this.worker = worker;
    }

    public get operatorType(): KubeflowOperator {
        return 'tf-operator';
    }
}

export class KubeflowTrialConfigPytorch extends KubeflowTrialConfig {
    public readonly master: KubernetesTrialConfigTemplate;
    public readonly worker?: KubernetesTrialConfigTemplate;

    constructor(codeDir: string, master: KubernetesTrialConfigTemplate, worker?: KubernetesTrialConfigTemplate) {
        super(codeDir);
        this.master = master;
        this.worker = worker;
    }

    public get operatorType(): KubeflowOperator {
        return 'pytorch-operator';
    }
}

export class KubeflowTrialConfigFactory {

    public static generateKubeflowTrialConfig(jsonObject: object, operator: KubeflowOperator): KubeflowTrialConfig {
        if(operator === 'tf-operator'){
            let kubeflowTrialConfigObject = <KubeflowTrialConfigTensorflow>jsonObject;
            return new KubeflowTrialConfigTensorflow(
                kubeflowTrialConfigObject.codeDir, 
                kubeflowTrialConfigObject.worker,
                kubeflowTrialConfigObject.ps
            );
        }else if(operator === 'pytorch-operator'){
            let kubeflowTrialConfigObject = <KubeflowTrialConfigPytorch>jsonObject;
            return new KubeflowTrialConfigPytorch(
                kubeflowTrialConfigObject.codeDir,
                kubeflowTrialConfigObject.master,
                kubeflowTrialConfigObject.worker
            );
        }
         throw new Error(`Invalid json object ${jsonObject}`);
    }
}
