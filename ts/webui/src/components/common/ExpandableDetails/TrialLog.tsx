import * as React from 'react';

interface TrialLogProps {
    logStr: string;
    logName: string;
}

const TrialLog = (props: TrialLogProps): any => {
    const { logStr, logName } = props;
    const isHyperlink = logStr.toLowerCase().startsWith('http');

    return (
        <div className='logpath'>
            <span className='logName'>{logName}</span>
            {isHyperlink ? (
                <a className='link' rel='noopener noreferrer' href={logStr} target='_blank'>
                    {logStr}
                </a>
            ) : (
                <span className='fontColor333'>{logStr}</span>
            )}
        </div>
    );
};

export default TrialLog;
