import { TrialConfig } from "../common/trialConfig";

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
export type KubeflowOperator = 'tf-operator' | 'pytorch-operator' | 'mxnet-operator' | 'caffe2-operator' | 'chainer-operator' | 'mpi-operator';
export type KubeflowOperatorPlural = 'tfjobs' | 'pytorchjobs' | 'mxjobs' | 'caffe2jobs' | 'chainerjobs' | 'mpijobs';

/**
 * map from Kubeflow operator name to its plural name in K8S
 */
export const kubeflowOperatorMap : Map<KubeflowOperator, KubeflowOperatorPlural> =  new Map<KubeflowOperator, KubeflowOperatorPlural>([
    ['tf-operator' , 'tfjobs'],
    ['pytorch-operator', 'pytorchjobs'],
    ['mxnet-operator', 'mxjobs'],
    ['caffe2-operator', 'caffe2jobs'],
    ['chainer-operator', 'chainerjobs'],
    ['mpi-operator', 'mpijobs']    
]);

/**
 * Kuberflow cluster configuration
 * 
 */
export class KubeflowClusterConfig {
    /** Name of Kubeflow operator, like tf-operator */
    public readonly operator: KubeflowOperator;
    public readonly nfs?: NFSConfig;
    public readonly kubernetesServer: string;
    public readonly keyVault?: keyVaultConfig;
    public readonly azureStorage?: AzureStorage;
    
    /**
     * Constructor
     * @param userName User name of Kubeflow Cluster
     * @param passWord password of Kubeflow Cluster
     * @param host Host IP of Kubeflow Cluster
     */
    constructor(operator: KubeflowOperator, kubernetesServer : string, nfs?: NFSConfig, keyVault?: keyVaultConfig, azureStorage ?: AzureStorage) {
        this.operator = operator;
        this.nfs = nfs;
        this.kubernetesServer = kubernetesServer;
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
export class KubeflowTrialConfig extends TrialConfig {
    public readonly cpuNum: number;
    public readonly memoryMB: number;
    public readonly image: string;
    
    constructor(command : string, codeDir : string, gpuNum : number, cpuNum: number, memoryMB: number, image: string) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
    }
}