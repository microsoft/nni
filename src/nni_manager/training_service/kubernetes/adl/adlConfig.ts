// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

/**
 * Checkpoint Config
 */
export class CheckpointConfig {
    public readonly storageClass: string;

    public readonly storageSize: string;

    constructor(storageClass: string, storageSize: string) {
        this.storageClass = storageClass;
        this.storageSize = storageSize;
    }
}

/**
 * imagePullSecret Config
 */
export class ImagePullSecretConfig{
    public readonly name: string;

    constructor(name: string) {
        this.name = name
    }
}

/**
 * NFS Config
 */
export class NFSConfig {
    public readonly server: string;

    public readonly path: string;

    public readonly containerMountPath: string;

    constructor(server: string, path: string, containerMountPath: string) {
        this.server = server;
        this.path = path;
        this.containerMountPath = containerMountPath;
    }
}

/**
 * Trial job configuration for Adl
 */
export class AdlTrialConfig {
    public readonly codeDir: string;

    public readonly command: string;

    public readonly gpuNum: number;

    public readonly image: string;

    public readonly imagePullSecrets?: ImagePullSecretConfig[];

    public readonly nfs?: NFSConfig;

    public readonly checkpoint?: CheckpointConfig;

    public readonly cpuNum?: number;

    public readonly memorySize?: string;

    constructor(codeDir: string,
                command: string, gpuNum: number,
                image: string, imagePullSecrets?: ImagePullSecretConfig[],
                nfs?: NFSConfig, checkpoint?: CheckpointConfig,
                cpuNum?: number, memorySize?: string) {
        this.codeDir = codeDir;
        this.command = command;
        this.gpuNum = gpuNum;
        this.image = image;
        this.imagePullSecrets = imagePullSecrets;
        this.nfs = nfs;
        this.checkpoint = checkpoint;
        this.cpuNum = cpuNum;
        this.memorySize = memorySize;
    }
}

export type AdlJobStatus = "Pending" | "Running" | "Starting" | "Stopping" | "Failed" | "Succeeded";
