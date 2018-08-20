export const MANAGER_IP = `${window.location.protocol}//${window.location.hostname}:51188/api/v1/nni`;
export const trialJobStatus = [
    'UNKNOWN',
    'WAITING',
    'RUNNING',
    'SUCCEEDED',
    'FAILED',
    'USER_CANCELED',
    'SYS_CANCELED'
];
export const CONTROLTYPE = [
    'SEARCH_SPACE',
    'TRIAL_CONCURRENCY',
    'MAX_EXEC_DURATION'
];
export const overviewItem = 5;
export const roundNum = (acc: number, n: number) => Math.round(acc * 10 ** n) / 10 ** n;
