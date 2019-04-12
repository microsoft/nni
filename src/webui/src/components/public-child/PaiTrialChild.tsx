import * as React from 'react';
import { Row } from 'antd';
import { DOWNLOAD_IP } from '../../static/const';

interface PaiTrialChildProps {
    logString: string;
    id: string;
    logCollect: boolean;
}

class PaiTrialChild extends React.Component<PaiTrialChildProps, {}> {

    constructor(props: PaiTrialChildProps) {
        super(props);

    }

    render() {
        const { logString, id, logCollect } = this.props;
        return (
            <div>
                {
                    logString === ''
                        ?
                        <div />
                        :
                        <Row>
                            {
                                logCollect
                                    ?
                                    <a
                                        target="_blank"
                                        href={`${DOWNLOAD_IP}/trial_${id}.log`}
                                        style={{ marginRight: 10 }}
                                    >
                                        trial stdout
                                    </a>
                                    :
                                    <span>trial stdout: {logString}</span>
                            }
                        </Row>
                }
            </div>
        );
    }
}

export default PaiTrialChild;
