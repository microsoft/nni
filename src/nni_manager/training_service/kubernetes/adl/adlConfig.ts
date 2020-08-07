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
 * Trial job configuration for Adl
 */
export class AdlTrialConfig {
    public readonly checkpoint: CheckpointConfig;

    public readonly codeDir: string;

    public readonly command: string;

    public readonly gpuNum: number;

    public readonly image: string;

    public readonly imagePullSecrets: ImagePullSecretConfig[];

    constructor(checkpoint: CheckpointConfig, codeDir: string,
                command: string, gpuNum: number,
                image: string, imagePullSecrets: ImagePullSecretConfig[]) {
        this.checkpoint = checkpoint;
        this.codeDir = codeDir;
        this.command = command;
        this.gpuNum = gpuNum;
        this.image = image;
        this.imagePullSecrets = imagePullSecrets;
    }
}

export type AdlJobStatus = "Pending" | "Running" | "Starting" | "Stopping" | "Failed" | "Succeeded";
