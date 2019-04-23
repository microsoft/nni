import * as React from 'react';
import {
    Row, Col,
    Tooltip
} from 'antd';
import { Experiment } from '../../static/interface';

interface BasicInfoProps {
    trialProfile: Experiment;
    status: string;
}

class BasicInfo extends React.Component<BasicInfoProps, {}> {

    constructor(props: BasicInfoProps) {
        super(props);
    }

    render() {
        const { trialProfile } = this.props;
        return (
            <Row className="main">
                <Col span={8} className="padItem basic">
                    <p>Name</p>
                    <div>{trialProfile.experName}</div>
                    <p>ID</p>
                    <div>{trialProfile.id}</div>
                </Col>
                <Col span={8} className="padItem basic">
                    <p>Start time</p>
                    <div className="nowrap">
                        {new Date(trialProfile.startTime).toLocaleString('en-US')}
                    </div>
                    <p>End time</p>
                    <div className="nowrap">
                        {
                            trialProfile.endTime
                                ?
                                new Date(trialProfile.endTime).toLocaleString('en-US')
                                :
                                'none'
                        }
                    </div>
                </Col>
                <Col span={8} className="padItem basic">
                    <p>Log directory</p>
                    <div className="nowrap">
                        <Tooltip placement="top" title={trialProfile.logDir}>
                            {trialProfile.logDir}
                        </Tooltip>
                    </div>
                    <p>Training platform</p>
                    <div className="nowrap">
                        {
                            trialProfile.trainingServicePlatform
                                ?
                                trialProfile.trainingServicePlatform
                                :
                                'none'
                        }
                    </div>
                </Col>
            </Row>
        );
    }
}

export default BasicInfo;