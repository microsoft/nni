import * as React from 'react';
import PropTypes from 'prop-types';
import TrialLog from './TrialLog';

const PaitrialLog = (props): any => {
    const { logStr } = props;
    const isHasNFSLog = logStr.indexOf(',') !== -1 ? true : false;
    return (
        <div>
            {isHasNFSLog ? (
                <div>
                    <TrialLog logStr={logStr.split(',')[0]} logName='Trial stdout:' />
                    <TrialLog logStr={logStr.split(',')[1]} logName='Log on NFS:' />
                </div>
            ) : (
                <TrialLog logStr={logStr} logName='Trial stdout:' />
            )}
        </div>
    );
};

PaitrialLog.propTypes = {
    logStr: PropTypes.string
};

export default PaitrialLog;
