import React from 'react';
import { Stack, TooltipHost, DirectionalHint } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { TOOLTIP_BACKGROUND_COLOR } from '../../../static/const';
import '../../../static/style/overview/command.scss';

export const Command = (): any => {
    const clusterMetaData = EXPERIMENT.profile.params.clusterMetaData;
    const tuner = EXPERIMENT.profile.params.tuner;
    const advisor = EXPERIMENT.profile.params.advisor;
    const assessor = EXPERIMENT.profile.params.assessor;
    let title = '';
    let builtinName = '';
    let trialCommand = 'unknown';
    if (tuner !== undefined) {
        title = title.concat('Tuner');
        if (tuner.builtinTunerName !== undefined) {
            builtinName = builtinName.concat(tuner.builtinTunerName);
        }
    }
    if (advisor !== undefined) {
        title = title.concat('/ Assessor');
        if (advisor.builtinAdvisorName !== undefined) {
            builtinName = builtinName.concat(advisor.builtinAdvisorName);
        }
    }
    if (assessor !== undefined) {
        title = title.concat('/ Addvisor');
        if (assessor.builtinAssessorName !== undefined) {
            builtinName = builtinName.concat(assessor.builtinAssessorName);
        }
    }
    if (clusterMetaData !== undefined) {
        for (const item of clusterMetaData) {
            if (item.key === 'command') {
                trialCommand = item.value;
            }
        }
    }
    return (
        <div className='command basic'>
            <div className='command1'>
                <p>Training platform</p>
                <div className='nowrap'>{EXPERIMENT.profile.params.trainingServicePlatform}</div>
                <p className='lineMargin'>{title}</p>
                <div className='nowrap'>{builtinName}</div>
            </div>
            <Stack className='command2'>
                <p>Log directory</p>
                <div className='nowrap'>
                    <TooltipHost
                        content={EXPERIMENT.profile.logDir || 'unknown'}
                        className='nowrap'
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
                        {EXPERIMENT.profile.logDir || 'unknown'}
                    </TooltipHost>
                </div>
                <p className='lineMargin'>Trial command</p>
                <div className='nowrap'>
                    <TooltipHost
                        content={trialCommand || 'unknown'}
                        className='nowrap'
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
                        {trialCommand || 'unknown'}
                    </TooltipHost>
                </div>
            </Stack>
        </div>
    );
};
