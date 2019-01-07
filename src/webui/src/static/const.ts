export const MANAGER_IP = `/api/v1/nni`;
export const DOWNLOAD_IP = `/logs`;
export const trialJobStatus = [
    'UNKNOWN',
    'WAITING',
    'RUNNING',
    'SUCCEEDED',
    'FAILED',
    'USER_CANCELED',
    'SYS_CANCELED',
    'EARLY_STOPPED'
];
export const CONTROLTYPE = [
    'SEARCH_SPACE',
    'TRIAL_CONCURRENCY',
    'MAX_EXEC_DURATION'
];
export const MONACO = {
    readOnly: true,
    automaticLayout: true
};
