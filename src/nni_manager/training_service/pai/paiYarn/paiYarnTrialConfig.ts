// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import {TrialConfig} from '../../common/trialConfig';

/**
 * PAI configuration to run trials
 */
export class PAIYarnTrialConfig extends TrialConfig {
    public readonly cpuNum: number;
    public readonly memoryMB: number;
    public readonly image: string;
    public readonly dataDir: string;
    public readonly outputDir: string;

    constructor(command: string, codeDir: string, gpuNum: number, cpuNum: number, memoryMB: number,
                image: string, dataDir: string, outputDir: string) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.dataDir = dataDir;
        this.outputDir = outputDir;
    }
}
