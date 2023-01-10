// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  TODO
 *  This module will handle the code discovery logic for 3rd-party training services.
 *  For now we only have "local_v3" and "remote_v3" as PoC.
 **/

import type { LocalConfig, RemoteConfig, TrainingServiceConfig } from 'common/experimentConfig';
import type { TrainingServiceV3 } from 'common/training_service_v3';
//import { LocalTrainingServiceV3 } from './local';
//import { RemoteTrainingServiceV3 } from './remote';

export function trainingServiceFactoryV3(config: TrainingServiceConfig): TrainingServiceV3 {
    //if (config.platform === 'local_v3') {
    //    return new LocalTrainingServiceV3(config);
    //} else if (config.platform === 'remote_v3') {
    //    return new RemoteTrainingServiceV3(config);
    //} else {
    throw new Error(`Bad training service platform: ${config.platform}`);
    //}
}
