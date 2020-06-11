// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as component from '../../../common/component';
import { KubernetesJobRestServer } from '../kubernetesJobRestServer';
import { KubeflowTrainingService } from './kubeflowTrainingService';

/**
 * Kubeflow Training service Rest server, provides rest API to support kubeflow job metrics update
 *
 */
@component.Singleton
export class KubeflowJobRestServer extends KubernetesJobRestServer {
    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super(component.get(KubeflowTrainingService));
    }
}
