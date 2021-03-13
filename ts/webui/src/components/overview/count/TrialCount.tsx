import * as React from 'react';
import { Stack, TooltipHost, ProgressIndicator, DirectionalHint } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '../../../static/datamodel';
import { CONTROLTYPE, TOOLTIP_BACKGROUND_COLOR, MAX_TRIAL_NUMBERS } from '../../../static/const';
import { EditExperimentParam } from './EditExperimentParam';
import { EditExpeParamContext } from './context';
import { ExpDurationContext } from './ExpDurationContext';
import { leftProgress, rightEidtParam, progressHeight } from './commonStyle';

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
                        <Stack horizontal className='ExpDuration'>
                            <div style={leftProgress}>
                                <TooltipHost
                                    content={`${bar2.toString()} trials`}
                                    directionalHint={DirectionalHint.bottomCenter}
                                    tooltipProps={{
                                        calloutProps: {
                                            styles: {
                                                beak: { background: TOOLTIP_BACKGROUND_COLOR },
                                                beakCurtain: { background: TOOLTIP_BACKGROUND_COLOR },
                                                calloutMain: { background: TOOLTIP_BACKGROUND_COLOR }
                                            }
                                        }
                                    }}
                                >
                                    <ProgressIndicator
                                        className={EXPERIMENT.status}
                                        percentComplete={bar2Percent}
                                        barHeight={progressHeight}
                                    />
                                </TooltipHost>
                                <div className='exp-progress'>
                                    <span className={`${EXPERIMENT.status} bold`}>{bar2}</span>
                                    <span className='joiner'>/</span>
                                    <span>{maxTrialNum}</span>
                                </div>
                            </div>
                        </Stack>
                        <Stack horizontal className='marginTop'>
                            <div style={leftProgress}>
                                <Stack horizontal className='status-count' gap={60}>
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
                                <Stack horizontal className='status-count marginTop' gap={80}>
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

                            <div style={rightEidtParam}>
                                <EditExpeParamContext.Provider
                                    value={{
                                        title: MAX_TRIAL_NUMBERS,
                                        field: 'maxTrialNum',
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
