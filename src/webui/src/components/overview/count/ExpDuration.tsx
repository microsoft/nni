import React from 'react';
import { Stack, TooltipHost, ProgressIndicator } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { CONTROLTYPE } from '../../../static/const';
import { convertDuration } from '../../../static/function';
import EditExperimentParam from './EditExperimentParam';
import {EditExpeParamContext} from './context';
import '../../../static/style/overview/count.scss';
const itemStyle1: React.CSSProperties = {
    width: '45%',
    height: 80
};
const itemStyle2: React.CSSProperties = {
    width: '55%',
    height: 80,
    textAlign: 'right',
    
};

export const ExpDuration = (): any => {

    const maxExecDuration = EXPERIMENT.profile.params.maxExecDuration;
    const percent = EXPERIMENT.profile.execDuration / maxExecDuration;
    const tooltip = maxExecDuration - EXPERIMENT.profile.execDuration;

    return (

        // <Stack horizontal horizontalAlign='space-between' className='ExpDuration'>
        <Stack horizontal className='ExpDuration'>
            <div style={itemStyle1}>
                <TooltipHost content={`${convertDuration(tooltip)} remaining`}>
                    <ProgressIndicator percentComplete={percent} barHeight={15} />
                </TooltipHost>
            </div>
            <div style={itemStyle2}>
                <Stack horizontal></Stack>
                <EditExpeParamContext.Provider value={{
                    editType: CONTROLTYPE[0], field: 'maxExecDuration', title: 'Max duration',
                    unit: 'min'
                }}>
                    <EditExperimentParam />
                </EditExpeParamContext.Provider>
            </div>

        </Stack>
    );
}
