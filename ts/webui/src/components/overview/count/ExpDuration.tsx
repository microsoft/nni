import React from 'react';
import { Stack, TooltipHost, ProgressIndicator } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { CONTROLTYPE } from '../../../static/const';
import { convertDuration } from '../../../static/function';
import { EditExperimentParam } from './EditExperimentParam';
import { ExpDurationContext } from './ExpDurationContext';
import { EditExpeParamContext } from './context';
import '../../../static/style/overview/count.scss';
const itemStyle1: React.CSSProperties = {
    width: '62%',
    height: 80
};
const itemStyle2: React.CSSProperties = {
    width: '63%',
    height: 80,
    textAlign: 'right'
};

export const ExpDuration = (): any => (
    <ExpDurationContext.Consumer>
        {(value): React.ReactNode => {
            const { maxExecDuration, execDuration, updateOverviewPage } = value;
            const tooltip = maxExecDuration - execDuration;
            const maxExecDurationStr = convertDuration(maxExecDuration);
            const percent = execDuration / maxExecDuration;
            return (
                <Stack horizontal className='ExpDuration'>
                    <div style={itemStyle1}>
                        <TooltipHost content={`${convertDuration(tooltip)} remaining`}>
                            <ProgressIndicator percentComplete={percent} barHeight={15} />
                        </TooltipHost>
                    </div>
                    <div style={itemStyle2}>
                        <Stack horizontal></Stack>
                        <EditExpeParamContext.Provider
                            value={{
                                editType: CONTROLTYPE[0],
                                field: 'maxExecDuration',
                                title: 'Max duration',
                                maxExecDuration: maxExecDurationStr,
                                maxTrialNum: EXPERIMENT.profile.params.maxTrialNum,
                                trialConcurrency: EXPERIMENT.profile.params.trialConcurrency,
                                updateOverviewPage
                            }}
                        >
                            <EditExperimentParam />
                        </EditExpeParamContext.Provider>
                    </div>
                </Stack>
            );
        }}
    </ExpDurationContext.Consumer>
);
