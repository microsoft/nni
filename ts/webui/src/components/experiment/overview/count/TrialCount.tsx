import * as React from 'react';
import { Stack, IStackTokens } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { CONTROLTYPE, MAX_TRIAL_NUMBERS } from '@static/const';
import { EditExperimentParam } from './EditExperimentParam';
import ProgressBar from './ProgressBar';
import { EditExpeParamContext } from './context';
import { ExpDurationContext } from './ExpDurationContext';
import { leftProgress, rightEditParam } from './commonStyle';

const line1Tokens: IStackTokens = {
    childrenGap: 60
};
const line2Tokens: IStackTokens = {
    childrenGap: 80
};

export const TrialCount = (): any => {
    const count = TRIALS.countStatus();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const stoppedCount = count.get('USER_CANCELED')! + count.get('SYS_CANCELED')! + count.get('EARLY_STOPPED')!;
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const bar2 = count.get('RUNNING')! + count.get('SUCCEEDED')! + count.get('FAILED')! + stoppedCount;
    const maxTrialNum = EXPERIMENT.maxTrialNumber;
    // support type [0, 1], not 98%
    const bar2Percent = bar2 / maxTrialNum;
    return (
        <ExpDurationContext.Consumer>
            {(value): React.ReactNode => {
                const { updateOverviewPage } = value;
                return (
                    <React.Fragment>
                        <ProgressBar
                            tooltip={`${bar2.toString()} trials`}
                            percent={bar2Percent}
                            latestVal={String(bar2)}
                            presetVal={String(maxTrialNum)}
                        />
                        <Stack horizontal className='marginTop'>
                            <div style={leftProgress}>
                                <Stack horizontal className='status-count' tokens={line1Tokens}>
                                    <div>
                                        <span>Running</span>
                                        <p>{count.get('RUNNING')}</p>
                                    </div>
                                    <div>
                                        <span>Succeeded</span>
                                        <p>{count.get('SUCCEEDED')}</p>
                                    </div>
                                    <div>
                                        <span>Stopped</span>
                                        <p>{stoppedCount}</p>
                                    </div>
                                </Stack>
                                <Stack horizontal className='status-count marginTop' tokens={line2Tokens}>
                                    <div>
                                        <span>Failed</span>
                                        <p>{count.get('FAILED')}</p>
                                    </div>
                                    <div>
                                        <span>Waiting</span>
                                        <p>{count.get('WAITING')}</p>
                                    </div>
                                </Stack>
                            </div>

                            <div style={rightEditParam}>
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
                                            // maxExecDuration: EXPERIMENT.profile.params.maxExecDuration,
                                            maxExecDuration: '',
                                            maxTrialNum: EXPERIMENT.maxTrialNumber,
                                            trialConcurrency: EXPERIMENT.profile.params.trialConcurrency,
                                            updateOverviewPage
                                        }}
                                    >
                                        <EditExperimentParam />
                                    </EditExpeParamContext.Provider>
                                </div>
                            </div>
                        </Stack>
                    </React.Fragment>
                );
            }}
        </ExpDurationContext.Consumer>
    );
};
