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
    public readonly paiStoragePlugin: string;
    public readonly paiConfigPath?: string;

    constructor(command: string, codeDir: string, gpuNum: number, cpuNum: number, memoryMB: number,
                image: string, nniManagerNFSMountPath: string, containerNFSMountPath: string,
                paiStoragePlugin: string, virtualCluster?: string, paiConfigPath?: string) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.virtualCluster = virtualCluster;
        this.nniManagerNFSMountPath = nniManagerNFSMountPath;
        this.containerNFSMountPath = containerNFSMountPath;
        this.paiStoragePlugin = paiStoragePlugin;
        this.paiConfigPath = paiConfigPath;
    }
}
