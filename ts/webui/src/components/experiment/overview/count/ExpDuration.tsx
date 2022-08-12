import React from 'react';
import { EXPERIMENT } from '@static/datamodel';
import { CONTROLTYPE } from '@static/const';
import { convertDuration, convertTimeAsUnit } from '@static/function';
import { EditExperimentParam } from './EditExperimentParam';
import { ExpDurationContext } from './ExpDurationContext';
import ProgressBar from './ProgressBar';
import { EditExpeParamContext } from './context';
import '@style/experiment/overview/count.scss';

export const ExpDuration = (): any => (
    <ExpDurationContext.Consumer>
        {(value): React.ReactNode => {
            const { maxExecDuration, execDuration, maxDurationUnit, updateOverviewPage } = value;
            const tooltip = maxExecDuration - execDuration;
            const percent = execDuration / maxExecDuration;
            const execDurationStr = convertDuration(execDuration);
            const maxExecDurationStr = convertTimeAsUnit(maxDurationUnit, maxExecDuration).toString();
            return (
                <React.Fragment>
                    <ProgressBar
                        tooltip={`${convertDuration(tooltip)} remaining`}
                        percent={percent}
                        latestVal={execDurationStr}
                        presetVal={`${maxExecDurationStr} ${maxDurationUnit}`}
                    />
                    <div className='editExpDuration'>
                        <EditExpeParamContext.Provider
                            value={{
                                editType: CONTROLTYPE[0],
                                field: 'maxExperimentDuration',
                                title: 'Max duration',
                                maxExecDuration: maxExecDurationStr,
                                maxTrialNum: EXPERIMENT.maxTrialNumber,
                                trialConcurrency: EXPERIMENT.profile.params.trialConcurrency,
                                updateOverviewPage
                            }}
                        >
                            <EditExperimentParam />
                        </EditExpeParamContext.Provider>
                    </div>
                </React.Fragment>
            );
        }}
    </ExpDurationContext.Consumer>
);
