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

import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';
// tslint:disable-next-line:no-implicit-dependencies
import * as request from 'request';
import * as component from '../../common/component';

import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import { MethodNotImplementedError } from '../../common/errors';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay, generateParamFileName,
    getExperimentRootDir, getIPV4Address, getVersion, uniqueString, unixPathJoin } from '../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir, execCopydir } from '../common/util';
import { PAILiteClusterConfig } from './paiLiteConfig';
import { PAI_LOG_PATH_FORMAT, PAI_TRIAL_COMMAND_FORMAT, PAITrialJobDetail } from './paiLiteData';
import { PAITrainingService } from '../pai/paiTrainingService';

import * as WebHDFS from 'webhdfs';

/**
 * Training Service implementation for OpenPAI (Open Platform for AI)
 * Refer https://github.com/Microsoft/pai for more info about OpenPAI
 */
@component.Singleton
class PAILiteTrainingService extends PAITrainingService {
    private containerRootDir: string = "";
    protected paiLiteClusterConfig?: PAILiteClusterConfig;

    constructor() {
        super();
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        console.log('---------------82-----------')
        const deferred : Deferred<PAITrialJobDetail> = new Deferred<PAITrialJobDetail>();
        return deferred.promise;
    }

    // tslint:disable:no-http-string
    public cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        return super.cancelTrialJob(trialJobId, isEarlyStopped);
    }

    // tslint:disable: no-unsafe-any no-any
    // tslint:disable-next-line:max-func-body-length
    public async setClusterMetadata(key: string, value: string): Promise<void> {
        const deferred : Deferred<void> = new Deferred<void>();
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                return super.setClusterMetadata(key, value);

            case TrialConfigMetadataKey.PAI_CLUSTER_CONFIG:
                deferred.resolve();
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:

                deferred.resolve();
                break;
            case TrialConfigMetadataKey.VERSION_CHECK:
                return super.setClusterMetadata(key, value);

            case TrialConfigMetadataKey.LOG_COLLECTION:
                return super.setClusterMetadata(key, value);

            case TrialConfigMetadataKey.MULTI_PHASE:
                return super.setClusterMetadata(key, value);

            default:
                //Reject for unknown keys
                deferred.reject(new Error(`Uknown key: ${key}`));
        }

        return deferred.promise;
    }

    // tslint:disable-next-line:max-func-body-length
    public async submitTrialJobToPAI(trialJobId: string): Promise<boolean> {
        const deferred : Deferred<boolean> = new Deferred<boolean>();
        return deferred.promise;
    }
}

export { PAILiteTrainingService };
