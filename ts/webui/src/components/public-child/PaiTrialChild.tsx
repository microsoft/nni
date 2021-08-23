import * as React from 'react';
import { DOWNLOAD_IP } from '../../static/const';
import LogPathChild from './LogPathChild';

interface PaiTrialChildProps {
    logString: string;
    id: string;
    logCollect: boolean;
}

class PaiTrialChild extends React.Component<PaiTrialChildProps, {}> {
    constructor(props: PaiTrialChildProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { logString, id, logCollect } = this.props;
        return (
            <div>
                {logString === '' ? null : (
                    <div>
                        {logCollect ? (
                            <a
                                target='_blank'
                                rel='noopener noreferrer'
                                href={`${DOWNLOAD_IP}/trial_${id}.log`}
                                style={{ marginRight: 10 }}
                            >
                                Trial stdout
                            </a>
                        ) : (
                            <LogPathChild eachLogpath={logString} logName='Trial stdout:' />
                        )}
                    </div>
                )}
            </div>
        );
    }
}

export default PaiTrialChild;
