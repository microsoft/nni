import React, { useContext } from 'react';
import { Stack, IStackTokens, ProgressIndicator } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { CONTROLTYPE, MAX_TRIAL_NUMBERS } from '@static/const';
import { EditExperimentParam } from './EditExperimentParam';
import { EditExpeParamContext } from './context';
import { AppContext } from '@/App';

const line1Tokens: IStackTokens = {
    childrenGap: 60
};
const editNumberConcurrency: IStackTokens = {
    childrenGap: 13
};
export const TrialCount = (): any => {
    const { updateOverviewPage } = useContext(AppContext);
    const count = TRIALS.countStatus();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const stoppedCount = count.get('USER_CANCELED')! + count.get('SYS_CANCELED')! + count.get('EARLY_STOPPED')!;
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const bar2 = count.get('RUNNING')! + count.get('SUCCEEDED')! + count.get('FAILED')! + stoppedCount;
    const maxTrialNum = EXPERIMENT.maxTrialNumber;
    // support type [0, 1], not 98%
    const bar2Percent = bar2 / maxTrialNum;
    return (
        <React.Fragment>
            <Stack horizontal horizontalAlign='space-between' tokens={line1Tokens} className='count marginTop'>
                <div>
                    <span className='font-untheme size16 count-succeed'>Succeeded</span>
                    <p className='size24 font-numbers-color succeed-trials-number'>{count.get('SUCCEEDED')}</p>

                    <ProgressIndicator
                        className={`${EXPERIMENT.status} fluent-progress`}
                        percentComplete={bar2Percent}
                        barHeight={8}
                    />
                    <div>
                        <span className='complete-tirals'>{bar2}</span>
                        <span className='font-untheme'>/{maxTrialNum}</span>
                    </div>
                </div>
                <div>
                    <div className='border run'></div>
                    <div className='font-untheme trial-status-style'>Running</div>
                    <p className='size18 font-numbers-color'>{count.get('RUNNING')}</p>

                    <div className='border stop marginTop'></div>
                    <div className='font-untheme trial-status-style'>Stopped</div>
                    <p className='size18 font-numbers-color'>{stoppedCount}</p>
                </div>
                <div className='numbers'>
                    <div className='border wait'></div>
                    <div className='font-untheme trial-status-style'>Waiting</div>
                    <p className='size18 font-numbers-color'>{count.get('WAITING')}</p>

                    <div className='border failed marginTop'></div>
                    <div className='font-untheme trial-status-style'>Failed</div>
                    <p className='size18 font-numbers-color'>{count.get('FAILED')}</p>
                </div>
            </Stack>
            <Stack horizontal className='edit-numbers' tokens={editNumberConcurrency}>
                <EditExpeParamContext.Provider
                    value={{
                        title: MAX_TRIAL_NUMBERS,
                        field: 'maxTrialNumber',
                        editType: CONTROLTYPE[1],
                        maxExecDuration: '',
                        maxTrialNum: EXPERIMENT.maxTrialNumber,
                        trialConcurrency: EXPERIMENT.profile.params.trialConcurrency,
                        updateOverviewPage
                    }}
                >
                    <div className='maxTrialNum'>
                        <EditExperimentParam />
                    </div>
                </EditExpeParamContext.Provider>
                <div className='concurrency'>
                    <EditExpeParamContext.Provider
                        value={{
                            title: 'Concurrency',
                            field: 'trialConcurrency',
                            editType: CONTROLTYPE[2],
                            maxExecDuration: '',
                            maxTrialNum: EXPERIMENT.maxTrialNumber,
                            trialConcurrency: EXPERIMENT.profile.params.trialConcurrency,
                            updateOverviewPage
                        }}
                    >
                        <EditExperimentParam />
                    </EditExpeParamContext.Provider>
                </div>
            </Stack>
        </React.Fragment>
    );
};
