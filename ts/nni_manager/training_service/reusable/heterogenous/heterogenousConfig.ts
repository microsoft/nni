// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


export class HeterogenousConfig {
    public readonly trainingServicePlatforms: string[];
    
    constructor(trainingServicePlatforms: string[]) {
        this.trainingServicePlatforms = trainingServicePlatforms;
    }
}
