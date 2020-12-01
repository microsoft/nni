// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as component from '../../../common/component';
import { KubernetesJobRestServer } from '../kubernetesJobRestServer';
import { AdlTrainingService } from './adlTrainingService';

/**
 * Adl Training service Rest server, provides rest API to support adl job metrics update
 *
 */
@component.Singleton
export class AdlJobRestServer extends KubernetesJobRestServer {
    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super(component.get(AdlTrainingService));
    }
}
