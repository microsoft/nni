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

/** operator types that kubeflow supported */
export type KubeflowOperator = 'tf-operator' | 'pytorch-operator' ;
export type KubeflowStorageKind = 'nfs' | 'azureStorage';
export type DistTrainRole = 'worker' | 'ps' | 'master';
export type OperatorApiVersion = 'v1alpha2' | 'v1beta1';

/**
 * Kuberflow cluster configuration
 * 
 */
export class KubeflowClusterConfigBase {
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

/**
 * NFS configuration to store Kubeflow job related files
 */
export class NFSConfig {
    /** IP Adress of NFS server */
    public readonly server : string;
    /** exported NFS path on NFS server */
    public readonly path : string;

    constructor(server : string, path : string) {
        this.server = server;
        this.path = path;
    }
}

/**
 * KeyVault configuration to store the key of Azure Storage Service
 * Refer https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2
 */
export class keyVaultConfig {
    /**The vault-name to specify vault */
    public readonly vaultName : string;
    /**The name to specify private key */
    public readonly name : string;

    constructor(vaultName : string, name : string){
        this.vaultName = vaultName;
        this.name = name;
    }
}

/**
 * Azure Storage Service
 */
export class AzureStorage {
    /**The azure share to storage files */
    public readonly azureShare : string;
    
    /**The account name of sotrage service */
    public readonly accountName: string;
    constructor(azureShare : string, accountName: string){
        this.azureShare = azureShare;
        this.accountName = accountName;
    }
}

/**
 * Trial job configuration for Kubeflow
 */
export class KubeflowTrialConfigTemplate {
    /** replication number of current role */
    public readonly replicas: number;

    /** CPU number */
    public readonly cpuNum: number;

    /** Memory  */
    public readonly memoryMB: number;

    /** Docker image */
    public readonly image: string;

    /** Trail command */
    public readonly command : string;

    /** Required GPU number for trial job. The number should be in [0,100] */
    public readonly gpuNum : number;
    
    constructor(replicas: number, command : string, gpuNum : number, 
        cpuNum: number, memoryMB: number, image: string) {
        this.replicas = replicas;
        this.command = command;
        this.gpuNum = gpuNum;
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
    }
}

export class KubeflowTrialConfigBase {
    public readonly codeDir: string;

    constructor(codeDir: string) {
        this.codeDir = codeDir;
    }
}

export class KubeflowTrialConfigTensorflow extends KubeflowTrialConfigBase{
    public readonly ps?: KubeflowTrialConfigTemplate;
    public readonly worker: KubeflowTrialConfigTemplate;

    constructor(codeDir: string, worker: KubeflowTrialConfigTemplate,  ps?: KubeflowTrialConfigTemplate) {
        super(codeDir);
        this.ps = ps;
        this.worker = worker;
    }
}

export class KubeflowTrialConfigPytorch extends KubeflowTrialConfigBase{
    public readonly master: KubeflowTrialConfigTemplate;
    public readonly worker?: KubeflowTrialConfigTemplate;

    constructor(codeDir: string, master: KubeflowTrialConfigTemplate, worker?: KubeflowTrialConfigTemplate) {
        super(codeDir);
        this.master = master;
        this.worker = worker;
    }
}

