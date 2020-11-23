import React from 'react';
import { TooltipHost, DirectionalHint } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { TOOLTIP_BACKGROUND_COLOR } from '../../../static/const';
import '../../../static/style/overview/command.scss';

export const Command2 = (): any => {
    const clusterMetaData = EXPERIMENT.profile.params.clusterMetaData;
    let trialCommand = 'unknown';

    if (clusterMetaData !== undefined) {
        for (const item of clusterMetaData) {
            if (item.key === 'command') {
                trialCommand = item.value as string;
            }
            if (item.key === 'trial_config') {
                if (typeof item.value === 'object' && 'command' in item.value) {
                    trialCommand = item.value.command as string;
                }
            }
        }
    }
    return (
        <div className='basic'>
            <p className='command'>Log directory</p>
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
        </div>
    );
};
