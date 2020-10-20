// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

export type KubernetesStorageKind = 'nfs' | 'azureStorage';
import { MethodNotImplementedError } from '../../common/errors';

export abstract class KubernetesClusterConfig {
    public readonly storage?: KubernetesStorageKind;
    public readonly apiVersion: string;

    constructor(apiVersion: string, storage?: KubernetesStorageKind) {
        this.storage = storage;
        this.apiVersion = apiVersion;
    }

    public get storageType(): KubernetesStorageKind {
        throw new MethodNotImplementedError();
    }
}

export class StorageConfig {
    public readonly storage?: KubernetesStorageKind;

    constructor(storage?: KubernetesStorageKind) {
        this.storage = storage;
    }
}

export class KubernetesClusterConfigNFS extends KubernetesClusterConfig {
    public readonly nfs: NFSConfig;

    constructor(
            apiVersion: string,
            nfs: NFSConfig,
            storage?: KubernetesStorageKind
        ) {
        super(apiVersion, storage);
        this.nfs = nfs;
    }

    public get storageType(): KubernetesStorageKind {
        return 'nfs';
    }

    public static getInstance(jsonObject: object): KubernetesClusterConfigNFS {
        const kubernetesClusterConfigObjectNFS: KubernetesClusterConfigNFS = <KubernetesClusterConfigNFS>jsonObject;

        return new KubernetesClusterConfigNFS(
            kubernetesClusterConfigObjectNFS.apiVersion,
            kubernetesClusterConfigObjectNFS.nfs,
            kubernetesClusterConfigObjectNFS.storage
        );
    }
}

export class KubernetesClusterConfigAzure extends KubernetesClusterConfig {
    public readonly keyVault: KeyVaultConfig;
    public readonly azureStorage: AzureStorage;
    public readonly uploadRetryCount: number | undefined;

    constructor(
            apiVersion: string,
            keyVault: KeyVaultConfig,
            azureStorage: AzureStorage,
            storage?: KubernetesStorageKind,
            uploadRetryCount?: number
        ) {
        super(apiVersion, storage);
        this.keyVault = keyVault;
        this.azureStorage = azureStorage;
        this.uploadRetryCount = uploadRetryCount;
    }

    public get storageType(): KubernetesStorageKind {
        return 'azureStorage';
    }

    public static getInstance(jsonObject: object): KubernetesClusterConfigAzure {
        const kubernetesClusterConfigObjectAzure: KubernetesClusterConfigAzure = <KubernetesClusterConfigAzure>jsonObject;

        return new KubernetesClusterConfigAzure(
            kubernetesClusterConfigObjectAzure.apiVersion,
            kubernetesClusterConfigObjectAzure.keyVault,
            kubernetesClusterConfigObjectAzure.azureStorage,
            kubernetesClusterConfigObjectAzure.storage,
            kubernetesClusterConfigObjectAzure.uploadRetryCount
        );
    }
}

export class KubernetesClusterConfigFactory {

    public static generateKubernetesClusterConfig(jsonObject: object): KubernetesClusterConfig {
         const storageConfig: StorageConfig = <StorageConfig>jsonObject;
         switch (storageConfig.storage) {
            case 'azureStorage':
                return KubernetesClusterConfigAzure.getInstance(jsonObject);
            case 'nfs':
            case undefined:
                return KubernetesClusterConfigNFS.getInstance(jsonObject);
            default:
                throw new Error(`Invalid json object ${jsonObject}`);
         }
    }
}

/**
 * NFS configuration to store Kubeflow job related files
 */
export class NFSConfig {
    // IP Adress of NFS server
    public readonly server: string;
    // exported NFS path on NFS server
    public readonly path: string;

    constructor(server: string, path: string) {
        this.server = server;
        this.path = path;
    }
}

/**
 * KeyVault configuration to store the key of Azure Storage Service
 * Refer https://docs.microsoft.com/en-us/azure/key-vault/key-vault-manage-with-cli2
 */
export class KeyVaultConfig {
    // The vault-name to specify vault
    public readonly vaultName: string;
    // The name to specify private key
    public readonly name: string;

    constructor(vaultName: string, name: string) {
        this.vaultName = vaultName;
        this.name = name;
    }
}

/**
 * Azure Storage Service
 */
export class AzureStorage {
    // The azure share to storage files
    public readonly azureShare: string;

    // The account name of sotrage service
    public readonly accountName: string;
    constructor(azureShare: string, accountName: string) {
        this.azureShare = azureShare;
        this.accountName = accountName;
    }
}

/**
 * Trial job configuration for Kubernetes
 */
export class KubernetesTrialConfigTemplate {
    // CPU number
    public readonly cpuNum: number;

    // Memory
    public readonly memoryMB: number;

    // Docker image
    public readonly image: string;

    // Private registry config file path to download docker iamge
    public readonly privateRegistryAuthPath?: string;

    // Trail command
    public readonly command: string;

    // Required GPU number for trial job. The number should be in [0,100]
    public readonly gpuNum: number;

    constructor(command: string, gpuNum: number,
                cpuNum: number, memoryMB: number, image: string, privateRegistryAuthPath?: string) {
        this.command = command;
        this.gpuNum = gpuNum;
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.privateRegistryAuthPath = privateRegistryAuthPath;
    }
}

export class KubernetesTrialConfig {
    public readonly codeDir: string;

    constructor(codeDir: string) {
        this.codeDir = codeDir;
    }
}
