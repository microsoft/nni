import * as React from 'react';
import { Stack, TooltipHost, ProgressIndicator } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '../../../static/datamodel';
import { CONTROLTYPE } from '../../../static/const';
import { EditExperimentParam } from './EditExperimentParam';
import { EditExpeParamContext } from './context';
import { ExpDurationContext } from './ExpDurationContext';

const itemStyles: React.CSSProperties = {
    width: '62%'
};

const itemStyle2: React.CSSProperties = {
    width: '63%',
    textAlign: 'right'
};

const itemStyle1: React.CSSProperties = {
    width: '30%',
    height: 50
};
const itemRunning: React.CSSProperties = {
    width: '42%',
    height: 56
};

export const TrialCount = (): any => {
    const count = TRIALS.countStatus();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const stoppedCount = count.get('USER_CANCELED')! + count.get('SYS_CANCELED')! + count.get('EARLY_STOPPED')!;
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const bar2 = count.get('RUNNING')! + count.get('SUCCEEDED')! + count.get('FAILED')! + stoppedCount;
    // support type [0, 1], not 98%
    const bar2Percent = bar2 / EXPERIMENT.profile.params.maxTrialNum;
    return (
        <ExpDurationContext.Consumer>
            {(value): React.ReactNode => {
                const { updateOverviewPage } = value;
                return (
                    <React.Fragment>
                        <Stack horizontal horizontalAlign='space-between' className='ExpDuration'>
                            <div style={itemStyles}>
                                <TooltipHost content={bar2.toString()}>
                                    <ProgressIndicator percentComplete={bar2Percent} barHeight={15} />
                                </TooltipHost>
                                <Stack horizontal className='mess'>
                                    <div style={itemRunning} className='basic colorOfbasic'>
                                        <p>Running</p>
                                        <div>{count.get('RUNNING')}</div>
                                    </div>
                                    <div style={itemStyle1} className='basic'>
                                        <p>Failed</p>
                                        <div>{count.get('FAILED')}</div>
                                    </div>
                                    <div style={itemStyle1} className='basic'>
                                        <p>Stopped</p>
                                        <div>{stoppedCount}</div>
                                    </div>
                                </Stack>
                                <Stack horizontal horizontalAlign='space-between' className='mess'>
                                    <div style={itemStyle1} className='basic colorOfbasic'>
                                        <p>Succeeded</p>
                                        <div>{count.get('SUCCEEDED')}</div>
                                    </div>

                                    <div style={itemStyle1} className='basic'>
                                        <p>Waiting</p>
                                        <div>{count.get('WAITING')}</div>
                                    </div>
                                </Stack>
                            </div>
                            <div style={itemStyle2}>
                                <EditExpeParamContext.Provider
                                    value={{
                                        title: 'Max trial numbers',
                                        field: 'maxTrialNum',
                                        editType: CONTROLTYPE[1],
                                        maxExecDuration: '',
                                        maxTrialNum: EXPERIMENT.profile.params.maxTrialNum,
                                        trialConcurrency: EXPERIMENT.profile.params.trialConcurrency,
                                        updateOverviewPage
                                    }}
                                >
                                    <EditExperimentParam />
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
                    </React.Fragment>
                );
            }}
        </ExpDurationContext.Consumer>
    );
};
