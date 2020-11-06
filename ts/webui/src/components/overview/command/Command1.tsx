import React from 'react';
import { EXPERIMENT } from '../../../static/datamodel';
import '../../../static/style/overview/command.scss';

export const Command1 = (): any => {
    const tuner = EXPERIMENT.profile.params.tuner;
    const advisor = EXPERIMENT.profile.params.advisor;
    const assessor = EXPERIMENT.profile.params.assessor;
    let title = '';
    let builtinName = '';
    if (tuner !== undefined) {
        title = 'Tuner';
        if (tuner.builtinTunerName !== undefined) {
            builtinName = builtinName.concat(tuner.builtinTunerName);
        }
    }

    if (advisor !== undefined) {
        title = 'Advisor';
        if (advisor.builtinAdvisorName !== undefined) {
            builtinName = builtinName.concat(advisor.builtinAdvisorName);
        }
        if (advisor.className !== undefined) {
            builtinName = builtinName.concat(advisor.className);
        }
    }

    if (assessor !== undefined) {
        title = title.concat('/ Assessor');
        if (assessor.builtinAssessorName !== undefined) {
            builtinName = builtinName.concat(assessor.builtinAssessorName);
        }
    }

    return (
        <div className='basic'>
            <div>
                <p className='command'>Training platform</p>
                <div className='nowrap'>{EXPERIMENT.profile.params.trainingServicePlatform}</div>
                <p className='lineMargin'>{title}</p>
                <div className='nowrap'>{builtinName}</div>
            </div>
        </div>
    );
};
