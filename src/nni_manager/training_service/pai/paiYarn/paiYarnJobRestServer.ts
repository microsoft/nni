// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Request, Response, Router } from 'express';
import { Inject } from 'typescript-ioc';
import * as component from '../../../common/component';
import { ClusterJobRestServer } from '../../common/clusterJobRestServer';
import { PAIYarnTrainingService } from './paiYarnTrainingService';
import { PAIJobRestServer } from '../paiJobRestServer';

export interface ParameterFileMeta {
    readonly experimentId: string;
    readonly trialId: string;
    readonly filePath: string;
}

/**
 * PAI Training service Rest server, provides rest API to support pai job metrics update
 *
 */
@component.Singleton
export class PAIYarnJobRestServer extends PAIJobRestServer {
    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super(component.get(PAIYarnTrainingService));
    }
}
