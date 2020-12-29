// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import {KubernetesTrialConfig} from "../kubernetesConfig";

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
export class AdlTrialConfig extends KubernetesTrialConfig {

    public readonly command: string;

    public readonly gpuNum: number;

    public readonly image: string;

    public readonly namespace?: string;

    public readonly imagePullSecrets?: ImagePullSecretConfig[];

    public readonly nfs?: NFSConfig;

    public readonly checkpoint?: CheckpointConfig;

    public readonly cpuNum?: number;

    public readonly memorySize?: string;

    public readonly adaptive?: boolean; // adaptive == preemptible

    constructor(codeDir: string,
                command: string, gpuNum: number,
                image: string, namespace?: string,
                imagePullSecrets?: ImagePullSecretConfig[],
                nfs?: NFSConfig, checkpoint?: CheckpointConfig,
                cpuNum?: number, memorySize?: string,
                adaptive?: boolean
    ) {
        super(codeDir);
        this.command = command;
        this.gpuNum = gpuNum;
        this.image = image;
        this.namespace = namespace;
        this.imagePullSecrets = imagePullSecrets;
        this.nfs = nfs;
        this.checkpoint = checkpoint;
        this.cpuNum = cpuNum;
        this.memorySize = memorySize;
        this.adaptive = adaptive;
    }
}

export type AdlJobStatus = "Pending" | "Running" | "Starting" | "Stopping" | "Failed" | "Succeeded";
