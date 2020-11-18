import React from 'react';
import { EXPERIMENT } from '../../../static/datamodel';
import '../../../static/style/overview/command.scss';
import '../../../static/style/common/ellipsis.scss';

export const Command1 = (): any => {
    const tuner = EXPERIMENT.profile.params.tuner;
    const advisor = EXPERIMENT.profile.params.advisor;
    const assessor = EXPERIMENT.profile.params.assessor;
    const title: string[] = [];
    const builtinName: string[] = [];
    if (tuner !== undefined) {
        title.push('Tuner');
        if (tuner.builtinTunerName !== undefined) {
            builtinName.push(tuner.builtinTunerName);
        }
    }

    if (advisor !== undefined) {
        title.push('Advisor');
        if (advisor.builtinAdvisorName !== undefined) {
            builtinName.push(advisor.builtinAdvisorName);
        }
        if (advisor.className !== undefined) {
            builtinName.push(advisor.className);
        }
    }

    if (assessor !== undefined) {
        title.push('Assessor');
        if (assessor.builtinAssessorName !== undefined) {
            builtinName.push(assessor.builtinAssessorName);
        }
    }

    return (
        <div className='basic'>
            <div>
                <p className='command'>Training platform</p>
                <div className='ellipsis'>{EXPERIMENT.profile.params.trainingServicePlatform}</div>
                <p className='lineMargin'>{title.join('/')}</p>
                <div className='ellipsis'>{builtinName.join('/')}</div>
            </div>
        </div>
    );
};
