import * as React from 'react';
import PropTypes from 'prop-types';

const TrialLog = (props): any => {
    const { logStr, logName } = props;
    const isHyperlink = /^http/gi.test(logStr);

    return (
        <div className='logpath'>
            <span className='logName'>{logName}</span>
            {isHyperlink ? (
                <a className='link' rel='noopener noreferrer' href={logStr} target='_blank'>
                    {logStr}
                </a>
            ) : (
                <span className='logContent'>{logStr}</span>
            )}
        </div>
    );
};

TrialLog.propTypes = {
    logStr: PropTypes.string,
    logName: PropTypes.string
};

export default TrialLog;
