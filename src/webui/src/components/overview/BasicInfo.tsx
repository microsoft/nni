import { Col, Row, Tooltip } from 'antd';
import * as React from 'react';
import { EXPERIMENT } from '../../static/datamodel';
import { formatTimestamp } from '../../static/function';

interface BasicInfoProps {
    experimentUpdateBroadcast: number;
}

class BasicInfo extends React.Component<BasicInfoProps, {}> {
    constructor(props: BasicInfoProps) {
        super(props);
    }

    render() {
        return (
            <Row className="main">
                <Col span={8} className="padItem basic">
                    <p>Name</p>
                    <div>{EXPERIMENT.profile.params.experimentName}</div>
                    <p>ID</p>
                    <div>{EXPERIMENT.profile.id}</div>
                </Col>
                <Col span={8} className="padItem basic">
                    <p>Start time</p>
                    <div className="nowrap">{formatTimestamp(EXPERIMENT.profile.startTime)}</div>
                    <p>End time</p>
                    <div className="nowrap">{formatTimestamp(EXPERIMENT.profile.endTime)}</div>
                </Col>
                <Col span={8} className="padItem basic">
                    <p>Log directory</p>
                    <div className="nowrap">
                        <Tooltip placement="top" title={EXPERIMENT.profile.logDir || ''}>
                            {EXPERIMENT.profile.logDir || 'unknown'}
                        </Tooltip>
                    </div>
                    <p>Training platform</p>
                    <div className="nowrap">{EXPERIMENT.profile.params.trainingServicePlatform}</div>
                </Col>
            </Row>
        );
    }
}

export default BasicInfo;
