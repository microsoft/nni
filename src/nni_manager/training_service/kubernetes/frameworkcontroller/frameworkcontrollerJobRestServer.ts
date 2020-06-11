// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as component from '../../../common/component';
import { KubernetesJobRestServer } from '../kubernetesJobRestServer';
import { FrameworkControllerTrainingService } from './frameworkcontrollerTrainingService';

/**
 * frameworkcontroller Training service Rest server, provides rest API to support frameworkcontroller job metrics update
 *
 */
@component.Singleton
export class FrameworkControllerJobRestServer extends KubernetesJobRestServer {
    constructor() {
        super(component.get(FrameworkControllerTrainingService));
    }
}
