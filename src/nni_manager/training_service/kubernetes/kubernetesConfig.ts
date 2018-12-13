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

export type KubernetesStorageKind = 'nfs' | 'azureStorage';

export class KubernetesClusterConfig {
    public readonly storage?: KubernetesStorageKind;
    
    constructor(storage?: KubernetesStorageKind) {
        this.storage = storage;
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
 * Trial job configuration for Kubernetes
 */
export class KubernetesTrialConfigTemplate {
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

export class KubernetesTrialConfig {
    public readonly codeDir: string;

    constructor(codeDir: string) {
        this.codeDir = codeDir;
    }
}