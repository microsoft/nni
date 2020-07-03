// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';
import {TrialConfig} from '../../common/trialConfig';

/**
 * PAI trial configuration
 */
export class NNIPAIK8STrialConfig extends TrialConfig {
    public readonly cpuNum: number;
    public readonly memoryMB: number;
    public readonly image: string;
    public virtualCluster?: string;
    public readonly nniManagerNFSMountPath: string;
    public readonly containerNFSMountPath: string;
    public readonly paiStorageConfigName: string;
    public readonly paiConfigPath?: string;

    constructor(command: string, codeDir: string, gpuNum: number, cpuNum: number, memoryMB: number,
                image: string, nniManagerNFSMountPath: string, containerNFSMountPath: string,
                paiStorageConfigName: string, virtualCluster?: string, paiConfigPath?: string) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.virtualCluster = virtualCluster;
        this.nniManagerNFSMountPath = nniManagerNFSMountPath;
        this.containerNFSMountPath = containerNFSMountPath;
        this.paiStorageConfigName = paiStorageConfigName;
        this.paiConfigPath = paiConfigPath;
    }
}
