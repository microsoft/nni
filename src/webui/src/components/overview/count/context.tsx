import React from 'react';
export const EditExpeParamContext = React.createContext({
    editType: '',
    field: '',
    title: '',
    maxExecDuration: '',
    maxTrialNum: 0,
    trialConcurrency: 0
});
/***
        * const CONTROLTYPE = ['MAX_EXEC_DURATION', 'MAX_TRIAL_NUM', 'TRIAL_CONCURRENCY', 'SEARCH_SPACE'];
        * [0], 'MAX_EXEC_DURATION', params.maxExecDuration
        * [1], 'MAX_TRIAL_NUM', params.maxTrialNum
        * [2], 'TRIAL_CONCURRENCY', params.trialConcurrency
        */
