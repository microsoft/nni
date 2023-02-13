import * as React from 'react';
import { Stack } from '@fluentui/react';
import { EXPERIMENT } from '@static/datamodel';
import { getPrefix } from '@static/function';
import TooltipHostIndex from '@components/common/TooltipHostIndex';

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
        <React.Fragment>
            <div className='bottomval'>
                <Stack horizontal className='experimentHead'>
                    <div className='small-icon'>
                        <img src={(getPrefix() || '') + '/icons/experiment-log-directory.png'} />
                    </div>
                    <div>
                        <p className='basic font-untheme'>Log directory</p>
                        <TooltipHostIndex value={EXPERIMENT.profile.logDir || 'unknown'} />
                    </div>
                </Stack>
            </div>
            <div className='bottomval'>
                <Stack horizontal className='experimentHead'>
                    <div className='small-icon'>
                        <img src={(getPrefix() || '') + '/icons/experiment-trial-command.png'} />
                    </div>
                    <div>
                        <p className='basic font-untheme'>Trial command</p>
                        <TooltipHostIndex value={EXPERIMENT.config.trialCommand || 'unknown'} />
                    </div>
                </Stack>
            </div>
            <div className='bottomval'>
                <Stack horizontal className='experimentHead'>
                    <div className='small-icon'>
                        <img src={(getPrefix() || '') + '/icons/experiment-platform.png'} />
                    </div>
                    <div>
                        <p className='basic font-untheme'>Training platform</p>
                        <div className='ellipsis name'>{EXPERIMENT.trainingServicePlatform}</div>
                    </div>
                </Stack>
            </div>
            <div className='bottomval'>
                <Stack horizontal className='experimentHead'>
                    <div className='small-icon'>
                        <img src={(getPrefix() || '') + '/icons/experiment-tuner.png'} />
                    </div>
                    <div>
                        <p className='basic font-untheme'>{title.join('/')}</p>
                        <div className='ellipsis name'>{builtinName.join('/')}</div>
                    </div>
                </Stack>
            </div>
        </React.Fragment>
    );
};

export default Config;
