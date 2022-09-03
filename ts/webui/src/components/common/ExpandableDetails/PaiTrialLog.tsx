import * as React from 'react';
import TrialLog from './TrialLog';

interface PaitrialLogProps {
    logStr: string;
}

const PaitrialLog = (props: PaitrialLogProps): any => {
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

export default PaitrialLog;
