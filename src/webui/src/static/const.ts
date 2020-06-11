// when there are more trials than this threshold, metrics will be updated in group of this size to avoid freezing
const METRIC_GROUP_UPDATE_THRESHOLD = 100;
const METRIC_GROUP_UPDATE_SIZE = 20;

let MANAGER_IP = `/api/v1/nni`;
if (process.env.NODE_ENV == "development") {
    MANAGER_IP = `//${window.location.hostname}:8080` + MANAGER_IP;
}
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
const CONTROLTYPE = [
    'SEARCH_SPACE',
    'TRIAL_CONCURRENCY',
    'MAX_EXEC_DURATION'
];
const MONACO = {
    readOnly: true,
    automaticLayout: true,
    scrollBeyondLastLine: false,
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
const COLUMNPro = ['Trial No.', 'ID', 'Start Time', 'End Time', 'Duration', 'Status',
    'Intermediate result', 'Default', OPERATION];
const CONCURRENCYTOOLTIP = 'Trial concurrency is the number of trials running concurrently.';

export {
    MANAGER_IP, DOWNLOAD_IP, trialJobStatus, COLUMNPro, WEBUIDOC,
    CONTROLTYPE, MONACO, COLUMN, DRAWEROPTION, OPERATION,
    METRIC_GROUP_UPDATE_THRESHOLD, METRIC_GROUP_UPDATE_SIZE, CONCURRENCYTOOLTIP
};
