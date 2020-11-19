// when there are more trials than this threshold, metrics will be updated in group of this size to avoid freezing
const METRIC_GROUP_UPDATE_THRESHOLD = 100;
const METRIC_GROUP_UPDATE_SIZE = 20;

const MANAGER_IP = `/api/v1/nni`;
const DOWNLOAD_IP = `/logs`;
const WEBUIDOC = 'https://nni.readthedocs.io/en/latest/Tutorial/WebUI.html';
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
// all choice column !dictory final
const COLUMNPro = [
    'Trial No.',
    'ID',
    'Start time',
    'End time',
    'Duration',
    'Status',
    'Intermediate result',
    'Default',
    OPERATION
];
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
const MAX_TRIAL_NUMBERS = 'Max trial No.';

export {
    MANAGER_IP,
    DOWNLOAD_IP,
    trialJobStatus,
    COLUMNPro,
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
    TOOLTIP_BACKGROUND_COLOR,
    MAX_TRIAL_NUMBERS
};
