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

    return (
        <div className='command basic'>
            <div>
                <p>Training platform</p>
                <div className='nowrap'>{EXPERIMENT.profile.params.trainingServicePlatform}</div>
                <p className='lineMargin'>{title}</p>
                <div className='nowrap'>{builtinName}</div>
            </div>
        </div>
    );
};
