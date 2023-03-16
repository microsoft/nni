import React, { useContext } from 'react';
import { Stack } from '@fluentui/react';
import { EXPERIMENT } from '@static/datamodel';
import { EditExpeParamContext } from './context';
import { EditExperimentParam } from './EditExperimentParam';
import { CONTROLTYPE } from '@static/const';
import { formatTimestamp, formatTimeStyle } from '@static/function';
import { convertTimeAsUnit } from '@static/function';
import { AppContext } from '@/App';
import { Circle } from 'rc-progress';
import '@style/experiment/overview/count.scss';

const filterStatus = (status: string): string => {
    switch (status) {
        case 'DONE':
        case 'STOPPED':
        case 'VIEWED':
            return '#00ad56';

        case 'ERROR':
            return '#a4262c';

        default:
            return '#0071bc';
    }
};

export const Duration = (): any => {
    const { maxDurationUnit, updateOverviewPage } = useContext(AppContext);
    const maxExecDuration = EXPERIMENT.maxExperimentDurationSeconds;
    const execDuration = EXPERIMENT.profile.execDuration;
    // because execDuration will be > maxExecDuration(it's by design) in experiment
    const percent = (execDuration / maxExecDuration) * 100 > 100 ? 100 : (execDuration / maxExecDuration) * 100;
    const maxExecDurationStr = convertTimeAsUnit(maxDurationUnit, maxExecDuration).toString();

    return (
        <React.Fragment>
            {/* duration progress circle */}
            <div className='circle-progress'>
                <Circle
                    className='progress'
                    percent={percent}
                    trailWidth={10}
                    strokeWidth={10}
                    strokeColor={filterStatus(EXPERIMENT.status)}
                    trailColor='#F3F5F7'
                    strokeLinecap='square'
                />
                <div
                    className='text duration-global-color'
                    dangerouslySetInnerHTML={{ __html: formatTimeStyle(execDuration) }}
                />
            </div>
            {/* max duration and edit max duration */}
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
            {/* start time and end time */}
            <Stack className='time' horizontal horizontalAlign='space-between'>
                <div>
                    <div className='font-untheme bg startMargin borderRadius'>Start</div>
                    <div className='font-untheme bg borderRadius'>End</div>
                </div>
                <div>
                    <div className='text startMargin'>
                        {formatTimestamp(EXPERIMENT.profile.startTime).split(',')[0]}
                    </div>
                    <div className='text'>{formatTimestamp(EXPERIMENT.profile.endTime).split(',')[0]}</div>
                </div>
                <div>
                    <div className='text startMargin'>
                        {formatTimestamp(EXPERIMENT.profile.startTime).split(',')[1]}
                    </div>
                    <div className='text'>{formatTimestamp(EXPERIMENT.profile.endTime).split(',')[1]}</div>
                </div>
            </Stack>
        </React.Fragment>
    );
};
