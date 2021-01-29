import React from 'react';
import { EXPERIMENT } from '../../../static/datamodel';
import { rightEidtParam } from '../count/commonStyle';
import '../../../static/style/overview/command.scss';

export const Command1 = (): any => {
    const tuner = EXPERIMENT.profile.params.tuner;
    const advisor = EXPERIMENT.profile.params.advisor;
    const assessor = EXPERIMENT.profile.params.assessor;
    const title: string[] = [];
    const builtinName: string[] = [];
    if (tuner !== undefined) {
        title.push('Tuner');
        builtinName.push(tuner.builtinTunerName || tuner.className || 'unknown');
    }

    if (advisor !== undefined) {
        title.push('Advisor');
        builtinName.push(advisor.builtinAdvisorName || advisor.className || 'unknown');
    }

    if (assessor !== undefined) {
        title.push('Assessor');
        builtinName.push(assessor.builtinAssessorName || assessor.className || 'unknown');
    }

    return (
        <div className='basic' style={rightEidtParam}>
            <div>
                <p className='command'>Training platform</p>
                <div className='ellipsis'>{EXPERIMENT.profile.params.trainingServicePlatform}</div>
                <p className='lineMargin'>{title.join('/')}</p>
                <div className='ellipsis'>{builtinName.join('/')}</div>
            </div>
        </div>
    );
};
