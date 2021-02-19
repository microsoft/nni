import React from 'react';
/***
 * const CONTROLTYPE = ['MAX_EXEC_DURATION', 'MAX_TRIAL_NUM', 'TRIAL_CONCURRENCY', 'SEARCH_SPACE'];
 * [0], 'MAX_EXEC_DURATION', params.maxExperimentDuration
 * [1], 'MAX_TRIAL_NUM', params.maxTrialNumber
 * [2], 'TRIAL_CONCURRENCY', params.trialConcurrency
 */
export const EditExpeParamContext = React.createContext({
    editType: '',
    field: '',
    title: '',
    maxExperimentDuration: '',
    maxTrialNumber: 0,
    trialConcurrency: 0,
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    updateOverviewPage: (): void => {}
});
