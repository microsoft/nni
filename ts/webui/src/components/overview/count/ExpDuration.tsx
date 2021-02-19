import React from 'react';
import { Stack, ProgressIndicator, TooltipHost, DirectionalHint } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { CONTROLTYPE, TOOLTIP_BACKGROUND_COLOR } from '../../../static/const';
import { convertDuration, convertTimeAsUnit } from '../../../static/function';
import { EditExperimentParam } from './EditExperimentParam';
import { ExpDurationContext } from './ExpDurationContext';
import { EditExpeParamContext } from './context';
import { leftProgress, rightEidtParam, progressHeight } from './commonStyle';
import '../../../static/style/overview/count.scss';

export const ExpDuration = (): any => (
    <ExpDurationContext.Consumer>
        {(value): React.ReactNode => {
            const { maxExperimentDuration, execDuration, maxDurationUnit, updateOverviewPage } = value;
            const tooltip = maxExperimentDuration - execDuration;
            const percent = execDuration / maxExperimentDuration;
            const execDurationStr = convertDuration(execDuration);
            const maxExecDurationStr = convertTimeAsUnit(maxDurationUnit, maxExperimentDuration).toString();
            return (
                <Stack horizontal className='ExpDuration'>
                    <div style={leftProgress}>
                        <TooltipHost
                            content={`${convertDuration(tooltip)} remaining`}
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
                                percentComplete={percent}
                                barHeight={progressHeight}
                            />
                        </TooltipHost>
                        {/* execDuration / maxDuration: 20min / 1h */}
                        <div className='exp-progress'>
                            <span className={`${EXPERIMENT.status} bold`}>{execDurationStr}</span>
                            <span className='joiner'>/</span>
                            <span>{`${maxExecDurationStr} ${maxDurationUnit}`}</span>
                        </div>
                    </div>
                    <div style={rightEidtParam}>
                        <EditExpeParamContext.Provider
                            value={{
                                editType: CONTROLTYPE[0],
                                field: 'maxExperimentDuration',
                                title: 'Max duration',
                                maxExperimentDuration: maxExecDurationStr,
                                maxTrialNumber: EXPERIMENT.profile.params.maxTrialNumber,
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
