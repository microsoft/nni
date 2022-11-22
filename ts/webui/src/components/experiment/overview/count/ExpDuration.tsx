import React, { useContext } from 'react';
import { EXPERIMENT } from '@static/datamodel';
import { CONTROLTYPE } from '@static/const';
import { convertDuration, convertTimeAsUnit } from '@static/function';
import { EditExperimentParam } from './EditExperimentParam';
import { AppContext } from '@/App';
import ProgressBar from './ProgressBar';
import { EditExpeParamContext } from './context';
import '@style/experiment/overview/count.scss';

export const ExpDuration = (): any => {
    const { maxDurationUnit, updateOverviewPage } = useContext(AppContext);
    const maxExecDuration = EXPERIMENT.maxExperimentDurationSeconds;
    const execDuration = EXPERIMENT.profile.execDuration;
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
}
