import React from 'react';
import { Stack, ProgressIndicator, TooltipHost, DirectionalHint } from '@fluentui/react';
import { EXPERIMENT } from '@static/datamodel';
import { TOOLTIPSTYLE } from '@static/const';
import { leftProgress, progressHeight } from './commonStyle';
import '@style/experiment/overview/count.scss';

interface ProgressBarProps {
    tooltip: string;
    percent: number;
    latestVal: string;
    presetVal: string;
}

const ProgressBar = (props: ProgressBarProps): any => {
    const { tooltip, percent, latestVal, presetVal } = props;
    return (
        <Stack horizontal className='marginTop'>
            <div style={leftProgress}>
                <TooltipHost
                    content={tooltip}
                    directionalHint={DirectionalHint.bottomCenter}
                    tooltipProps={TOOLTIPSTYLE}
                >
                    <ProgressIndicator
                        className={EXPERIMENT.status}
                        percentComplete={percent}
                        barHeight={progressHeight}
                    />
                </TooltipHost>
                <div className='exp-progress'>
                    <span className={`${EXPERIMENT.status} bold`}>{latestVal}</span>
                    <span className='joiner'>/</span>
                    <span>{presetVal}</span>
                </div>
            </div>
        </Stack>
    );
};

export default ProgressBar;
