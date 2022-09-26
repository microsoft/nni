import * as React from 'react';
import { Stack } from '@fluentui/react';
import { EXPERIMENT } from '@static/datamodel';
import { leftProgress, rightEditParam } from './count/commonStyle';
import TooltipHostIndex from '@components/common/TooltipHostIndex';
import '@style/experiment/overview/command.scss';

// This file is for showing the experiment some important params,
// Log directory, trial command, training platform and tuner/advisor message

const Config = (): any => {
    const tuner = EXPERIMENT.profile.params.tuner;
    const advisor = EXPERIMENT.profile.params.advisor;
    const assessor = EXPERIMENT.profile.params.assessor;
    const title: string[] = [];
    const builtinName: string[] = [];
    if (tuner !== undefined) {
        title.push('Tuner');
        builtinName.push(tuner.name || tuner.className || 'unknown');
    }

    if (advisor !== undefined) {
        title.push('Advisor');
        builtinName.push(advisor.name || advisor.className || 'unknown');
    }

    if (assessor !== undefined) {
        title.push('Assessor');
        builtinName.push(assessor.name || assessor.className || 'unknown');
    }

    return (
        <Stack horizontal>
            <div className='basic' style={leftProgress}>
                <p className='command'>Log directory</p>
                <TooltipHostIndex value={EXPERIMENT.profile.logDir || 'unknown'} />
                <p className='lineMargin'>Trial command</p>
                <TooltipHostIndex value={EXPERIMENT.config.trialCommand || 'unknown'} />
            </div>
            <div className='basic' style={rightEditParam}>
                <div>
                    <p className='command'>Training platform</p>
                    <div className='ellipsis'>{EXPERIMENT.trainingServicePlatform}</div>
                    <p className='lineMargin'>{title.join('/')}</p>
                    <div className='ellipsis'>{builtinName.join('/')}</div>
                </div>
            </div>
        </Stack>
    );
};

export default Config;
