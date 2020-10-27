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
    REMOTE_CONFIG = 'remote_config',
    EXPERIMENT_ID = 'experimentId',
    MULTI_PHASE = 'multiPhase',
    RANDOM_SCHEDULER = 'random_scheduler',
    PAI_YARN_CLUSTER_CONFIG = 'pai_yarn_config',
    PAI_CLUSTER_CONFIG = 'pai_config',
    KUBEFLOW_CLUSTER_CONFIG = 'kubeflow_config',
    NNI_MANAGER_IP = 'nni_manager_ip',
    FRAMEWORKCONTROLLER_CLUSTER_CONFIG = 'frameworkcontroller_config',
    DLTS_CLUSTER_CONFIG = 'dlts_config',
    AML_CLUSTER_CONFIG = 'aml_config',
    VERSION_CHECK = 'version_check',
    LOG_COLLECTION = 'log_collection'
}
