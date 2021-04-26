import React from 'react';
import { TooltipHost, DirectionalHint } from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { leftProgress } from '../count/commonStyle';
import { TOOLTIP_BACKGROUND_COLOR } from '../../../static/const';
import '../../../static/style/overview/command.scss';

export const Command2 = (): any => {
    return (
        <div className='basic' style={leftProgress}>
            <p className='command'>Log directory</p>
            <div className='ellipsis'>
                <TooltipHost
                    content={EXPERIMENT.profile.logDir || 'unknown'}
                    className='ellipsis'
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
            <div className='ellipsis'>
                <TooltipHost
                    content={EXPERIMENT.config.trialCommand || 'unknown'}
                    className='ellipsis'
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
                    {EXPERIMENT.config.trialCommand || 'unknown'}
                </TooltipHost>
            </div>
        </div>
    );
};
