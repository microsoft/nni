import { getPrefix } from './function';

// when there are more trials than this threshold, metrics will be updated in group of this size to avoid freezing
const METRIC_GROUP_UPDATE_THRESHOLD = 100;
const METRIC_GROUP_UPDATE_SIZE = 20;

/**
 *  RESTAPI and DOWNLOAD_IP must be synchronized with:
 *    - nni/experiment/rest.py
 *    - ts/nni_manager/rest_server/index.ts
 *  Remember to update them if the values are changed or if this file is moved.
 **/

const prefix = getPrefix();
const RESTAPI = '/api/v1/nni';
const MANAGER_IP = prefix === undefined ? RESTAPI : `${prefix}${RESTAPI}`;
const DOWNLOAD_IP = prefix === undefined ? '/logs' : `${prefix}/logs`;

const WEBUIDOC = 'https://nni.readthedocs.io/en/latest/experiment/webui.html';

const trialJobStatus = [
    'UNKNOWN',
    'WAITING',
    'RUNNING',
    'SUCCEEDED',
    'FAILED',
    'USER_CANCELED',
    'SYS_CANCELED',
    'EARLY_STOPPED'
];
const EXPERIMENTSTATUS = [
    'INITIALIZED',
    'RUNNING',
    'ERROR',
    'STOPPING',
    'STOPPED',
    'VIEWED',
    'DONE',
    'NO_MORE_TRIAL',
    'TUNER_NO_MORE_TRIAL'
];
const CONTROLTYPE = ['MAX_EXEC_DURATION', 'MAX_TRIAL_NUM', 'TRIAL_CONCURRENCY', 'SEARCH_SPACE'];
const MONACO = {
    readOnly: true,
    automaticLayout: true,
    scrollBeyondLastLine: false
};
const DRAWEROPTION = {
    minimap: { enabled: false },
    readOnly: true,
    automaticLayout: true
};
const OPERATION = 'Operation';
// defatult selected column
const COLUMN = ['Trial No.', 'ID', 'Duration', 'Status', 'Default', OPERATION];
const CONCURRENCYTOOLTIP = 'Trial concurrency is the number of trials running concurrently.';
const SUPPORTED_SEARCH_SPACE_TYPE = [
    'choice',
    'layer_choice',
    'input_choice',
    'randint',
    'uniform',
    'quniform',
    'loguniform',
    'qloguniform',
    'normal',
    'qnormal',
    'lognormal',
    'qlognormal'
];

const TOOLTIP_BACKGROUND_COLOR = '#484848';
const TOOLTIPSTYLE = {
    calloutProps: {
        styles: {
            beak: { background: TOOLTIP_BACKGROUND_COLOR },
            beakCurtain: { background: TOOLTIP_BACKGROUND_COLOR },
            calloutMain: { background: TOOLTIP_BACKGROUND_COLOR }
        }
    }
};
const MAX_TRIAL_NUMBERS = 'Max trial No.';
const RETIARIIPARAMETERS = 'mutation_summary';

export {
    MANAGER_IP,
    DOWNLOAD_IP,
    trialJobStatus,
    EXPERIMENTSTATUS,
    WEBUIDOC,
    CONTROLTYPE,
    MONACO,
    COLUMN,
    DRAWEROPTION,
    OPERATION,
    METRIC_GROUP_UPDATE_THRESHOLD,
    METRIC_GROUP_UPDATE_SIZE,
    CONCURRENCYTOOLTIP,
    SUPPORTED_SEARCH_SPACE_TYPE,
    TOOLTIPSTYLE,
    MAX_TRIAL_NUMBERS,
    RETIARIIPARAMETERS
};
