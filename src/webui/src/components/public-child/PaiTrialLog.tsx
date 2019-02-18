import * as React from 'react';
import { Row } from 'antd';
import { DOWNLOAD_IP } from '../../static/const';
import PaiTrialChild from './PaiTrialChild';

interface PaitrialLogProps {
    logStr: string;
    id: string;
    showLogModal: Function;
    trialStatus?: string;
    isdisLogbutton?: boolean;
}

class PaitrialLog extends React.Component<PaitrialLogProps, {}> {

    constructor(props: PaitrialLogProps) {
        super(props);

    }

    render() {
        const { logStr, id, showLogModal, 
            isdisLogbutton 
        } = this.props;
        const isTwopath = logStr.indexOf(',') !== -1
            ?
            true
            :
            false;
        return (
            <div>
                <div>
                    {
                        isTwopath
                            ?
                            <Row>
                                <Row>
                                    <a
                                        target="_blank"
                                        href={`${DOWNLOAD_IP}/trial_${id}.log`}
                                        style={{ marginRight: 10 }}
                                    >
                                        trial stdout
                                    </a>
                                    <a target="_blank" href={logStr.split(',')[1]}>hdfsLog</a>
                                </Row>
                            </Row>
                            :
                            <PaiTrialChild
                                logString={logStr}
                                id={id}
                                showLogModal={showLogModal}
                                isdisLogbtn={isdisLogbutton}
                            />
                    }
                </div>
            </div>
        );
    }
}

export default PaitrialLog;
