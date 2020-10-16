import * as React from 'react';
import { Stack, TooltipHost, ProgressIndicator, DirectionalHint } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '../../../static/datamodel';
import { CONTROLTYPE, TOOLTIP_BACKGROUND_COLOR, MAX_TRIAL_NUMBERS } from '../../../static/const';
import { EditExperimentParam } from './EditExperimentParam';
import { EditExpeParamContext } from './context';
import { ExpDurationContext } from './ExpDurationContext';
import { itemStyle1, itemStyle2 } from './commonStyle';

export const TrialCount = (): any => {
    const count = TRIALS.countStatus();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const stoppedCount = count.get('USER_CANCELED')! + count.get('SYS_CANCELED')! + count.get('EARLY_STOPPED')!;
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const bar2 = count.get('RUNNING')! + count.get('SUCCEEDED')! + count.get('FAILED')! + stoppedCount;
    const maxTrialNum = EXPERIMENT.profile.params.maxTrialNum;
    // support type [0, 1], not 98%
    const bar2Percent = bar2 / maxTrialNum;
    return (
        <ExpDurationContext.Consumer>
            {(value): React.ReactNode => {
                const { updateOverviewPage } = value;
                return (
                    <React.Fragment>
                        <Stack horizontal horizontalAlign='space-between' className='ExpDuration'>
                            <div style={itemStyle1}>
                                <TooltipHost
                                    content={bar2.toString()}
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
                                        barHeight={15}
                                    />
                                </TooltipHost>
                                <div className='exp-progress'>
                                    <span className={`${EXPERIMENT.status} bold`}>{bar2}</span>
                                    <span className='joiner'>/</span>
                                    <span>{maxTrialNum}</span>
                                </div>
                            </div>
                            <div style={itemStyle2}>
                                <EditExpeParamContext.Provider
                                    value={{
                                        title: MAX_TRIAL_NUMBERS,
                                        field: 'maxTrialNum',
                                        editType: CONTROLTYPE[1],
                                        maxExecDuration: '',
                                        maxTrialNum: EXPERIMENT.profile.params.maxTrialNum,
                                        trialConcurrency: EXPERIMENT.profile.params.trialConcurrency,
                                        updateOverviewPage
                                    }}
                                >
                                    <div className='maxTrialNum'>
                                        <EditExperimentParam />
                                    </div>
                                </EditExpeParamContext.Provider>
                                <EditExpeParamContext.Provider
                                    value={{
                                        title: 'Concurrency',
                                        field: 'trialConcurrency',
                                        editType: CONTROLTYPE[2],
                                        // maxExecDuration: EXPERIMENT.profile.params.maxExecDuration,
                                        maxExecDuration: '',
                                        maxTrialNum: EXPERIMENT.profile.params.maxTrialNum,
                                        trialConcurrency: EXPERIMENT.profile.params.trialConcurrency,
                                        updateOverviewPage
                                    }}
                                >
                                    <EditExperimentParam />
                                </EditExpeParamContext.Provider>
                            </div>
                        </Stack>
                        <Stack horizontal horizontalAlign='space-between' className='mess'>
                            <div className='basic'>
                                <p>Running</p>
                                <div>{count.get('RUNNING')}</div>
                            </div>
                            <div className='basic'>
                                <p>Failed</p>
                                <div>{count.get('FAILED')}</div>
                            </div>
                            <div className='basic'>
                                <p>Stopped</p>
                                <div>{stoppedCount}</div>
                            </div>
                            <div className='basic'>
                                <p>Succeeded</p>
                                <div>{count.get('SUCCEEDED')}</div>
                            </div>
                            <div className='basic'>
                                <p>Waiting</p>
                                <div>{count.get('WAITING')}</div>
                            </div>
                        </Stack>
                    </React.Fragment>
                );
            }}
        </ExpDurationContext.Consumer>
    );
};
