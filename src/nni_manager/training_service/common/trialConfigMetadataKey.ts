// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

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
