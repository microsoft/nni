// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

/**
 * Trial job configuration class
 * Representing trial job configurable properties
 */
export class TrialConfig {
    // Trail command
    public readonly command: string;

    // Code directory
    public readonly codeDir: string;

    // Required GPU number for trial job. The number should be in [0,100]
    public readonly gpuNum: number;

    // this flag uses for UT now.
    // in future, all environments should be reusable, and this can be configurable by user.
    public reuseEnvironment: boolean | undefined = true;

    /**
     * Constructor
     * @param command Trail command
     * @param codeDir Code directory
     * @param gpuNum Required GPU number for trial job
     */
    constructor(command: string, codeDir: string, gpuNum: number) {
        this.command = command;
        this.codeDir = codeDir;
        this.gpuNum = gpuNum;
    }
}
