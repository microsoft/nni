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

/**
 * Enum of metadata keys for configuration
 */
export enum TrialConfigMetadataKey {
    MACHINE_LIST = 'machine_list',
    LOCAL_CONFIG = 'local_config',
    TRIAL_CONFIG = 'trial_config',
    EXPERIMENT_ID = 'experimentId',
    MULTI_PHASE = 'multiPhase',
    RANDOM_SCHEDULER = 'random_scheduler',
    PAI_CLUSTER_CONFIG = 'pai_config',
    KUBEFLOW_CLUSTER_CONFIG = 'kubeflow_config',
    NNI_MANAGER_IP = 'nni_manager_ip',
    FRAMEWORKCONTROLLER_CLUSTER_CONFIG = 'frameworkcontroller_config',
    VERSION_CHECK = 'version_check',
    LOG_COLLECTION = 'log_collection'
}
