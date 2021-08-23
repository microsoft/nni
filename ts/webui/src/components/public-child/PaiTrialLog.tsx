import * as React from 'react';
import { DOWNLOAD_IP } from '../../static/const';
import PaiTrialChild from './PaiTrialChild';
import LogPathChild from './LogPathChild';

interface PaitrialLogProps {
    logStr: string;
    id: string;
    logCollection: boolean;
}

class PaitrialLog extends React.Component<PaitrialLogProps, {}> {
    constructor(props: PaitrialLogProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { logStr, id, logCollection } = this.props;
        const isTwopath = logStr.indexOf(',') !== -1 ? true : false;
        return (
            <div>
                <div>
                    {isTwopath ? (
                        <div>
                            {logCollection ? (
                                <div>
                                    <a
                                        target='_blank'
                                        rel='noopener noreferrer'
                                        href={`${DOWNLOAD_IP}/trial_${id}.log`}
                                        style={{ marginRight: 10 }}
                                    >
                                        Trial stdout
                                    </a>
                                    <a target='_blank' rel='noopener noreferrer' href={logStr.split(',')[1]}>
                                        NFS log
                                    </a>
                                </div>
                            ) : (
                                <div>
                                    <LogPathChild eachLogpath={logStr.split(',')[0]} logName='Trial stdout:' />
                                    <LogPathChild eachLogpath={logStr.split(',')[1]} logName='Log on NFS:' />
                                </div>
                            )}
                        </div>
                    ) : (
                        <PaiTrialChild logString={logStr} id={id} logCollect={logCollection} />
                    )}
                </div>
            </div>
        );
    }
}

export default PaitrialLog;
