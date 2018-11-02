import * as React from 'react';
import {
    Row,
    Col,
} from 'antd';
import { Experiment, TrialNumber } from '../../static/interface';
import { convertTime } from '../../static/function';
import ProgressBar from './ProgressItem';
import '../../static/style/progress.scss';

interface ProgressProps {
    trialProfile: Experiment;
    trialNumber: TrialNumber;
    bestAccuracy: string;
    status: string;
}

class Progressed extends React.Component<ProgressProps, {}> {

    constructor(props: ProgressProps) {
        super(props);
    }

    render() {
        const { trialProfile,
            trialNumber, bestAccuracy,
            status
        } = this.props;
        // remaining time
        const bar2 = trialNumber.totalCurrentTrial - trialNumber.waitTrial - trialNumber.unknowTrial;
        const bar2Percent = (bar2 / trialProfile.MaxTrialNum) * 100;
        const percent = (trialProfile.execDuration / trialProfile.maxDuration) * 100;
        const runDuration = convertTime(trialProfile.execDuration);
        const remaining = convertTime(trialProfile.maxDuration - trialProfile.execDuration);

        return (
            <Row className="progress">
                <Row className="basic">
                    <p>Status</p>
                    <div className="status">{status}</div>
                </Row>
                <ProgressBar
                    who="Duration"
                    percent={percent}
                    description={runDuration}
                    maxString={`MaxDuration: ${convertTime(trialProfile.maxDuration)}`}
                />
                <ProgressBar
                    who="TrialNum"
                    percent={bar2Percent}
                    description={bar2.toString()}
                    maxString={`MaxTrialNumber: ${trialProfile.MaxTrialNum}`}
                />
                <Row className="basic colorOfbasic mess">
                    <p>Best Accuracy</p>
                    <div>{bestAccuracy}</div>
                </Row>
                <Row className="mess">
                    <Col span={8}>
                        <Row className="basic colorOfbasic">
                            <p>Time Spent</p>
                            <div>{convertTime(trialProfile.execDuration)}</div>
                        </Row>
                    </Col>
                    <Col span={9}>
                        <Row className="basic colorOfbasic">
                            <p>Remaining Time</p>
                            <div>{remaining}</div>
                        </Row>
                    </Col>
                    <Col span={7}>
                        <Row className="basic colorOfbasic">
                            <p>Duration</p>
                            <div>{convertTime(trialProfile.maxDuration)}</div>
                        </Row>
                    </Col>
                </Row>
                <Row className="mess">
                    <Col span={8}>
                        <Row className="basic colorOfbasic">
                            <p>Succeed Trial</p>
                            <div>{trialNumber.succTrial}</div>
                        </Row>
                    </Col>
                    <Col span={9}>
                        <Row className="basic">
                            <p>Stopped Trial</p>
                            <div>{trialNumber.stopTrial}</div>
                        </Row>
                    </Col>
                    <Col span={7}>
                        <Row className="basic">
                            <p>Failed Trial</p>
                            <div>{trialNumber.failTrial}</div>
                        </Row>
                    </Col>
                </Row>
            </Row>
        );
    }
}

export default Progressed;