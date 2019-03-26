const MANAGER_IP = `/api/v1/nni`;
const DOWNLOAD_IP = `/logs`;
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
    automaticLayout: true
};
const COLUMN_INDEX = [
    {
        name: 'Trial No.',
        index: 1
    },
    {
        name: 'Id',
        index: 2
    },
    {
        name: 'Duration',
        index: 3
    },
    {
        name: 'Status',
        index: 4
    },
    {
        name: 'Default',
        index: 5
    },
    {
        name: 'Operation',
        index: 10000
    },
    {
        name: 'Intermediate Result',
        index: 10001
    }
];
const COLUMN = ['Trial No.', 'Id', 'Duration', 'Status', 'Default', 'Operation', 'Intermediate Result'];
export {
    MANAGER_IP, DOWNLOAD_IP, trialJobStatus,
    CONTROLTYPE, MONACO, COLUMN, COLUMN_INDEX
};
