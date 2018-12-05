import * as React from 'react';
import {
    Row,
    Col,
    Popover
} from 'antd';
import { Experiment, TrialNumber } from '../../static/interface';
import { convertTime } from '../../static/function';
import ProgressBar from './ProgressItem';
import '../../static/style/progress.scss';
import '../../static/style/probar.scss';

interface ProgressProps {
    trialProfile: Experiment;
    trialNumber: TrialNumber;
    bestAccuracy: string;
    status: string;
    errors: string;
}

class Progressed extends React.Component<ProgressProps, {}> {

    constructor(props: ProgressProps) {
        super(props);
    }

    render() {
        const { trialProfile,
            trialNumber, bestAccuracy,
            status, errors
        } = this.props;
        const bar2 = trialNumber.totalCurrentTrial - trialNumber.waitTrial - trialNumber.unknowTrial;
        const bar2Percent = (bar2 / trialProfile.MaxTrialNum) * 100;
        const percent = (trialProfile.execDuration / trialProfile.maxDuration) * 100;
        const runDuration = convertTime(trialProfile.execDuration);
        let remaining;
        if (status === 'DONE') {
            remaining = '0';
        } else {
            remaining = convertTime(trialProfile.maxDuration - trialProfile.execDuration);
        }
        let errorContent;
        if (errors !== '') {
            errorContent = (
                <div className="errors">
                    {errors}
                </div>
            );
        }
        return (
            <Row className="progress" id="barBack">
                <Row className="basic">
                    <p>Status</p>
                    <div className="status">
                        <span className={status}>{status}</span>
                        {
                            status === 'ERROR'
                                ?
                                <Popover
                                    placement="rightTop"
                                    content={errorContent}
                                    title="Error"
                                    trigger="hover"
                                >
                                    <span className="errorBtn">i</span>
                                </Popover>
                                :
                                <span />
                        }

                    </div>
                </Row>
                <ProgressBar
                    who="Duration"
                    percent={percent}
                    description={runDuration}
                    bgclass={status}
                    maxString={`MaxDuration: ${convertTime(trialProfile.maxDuration)}`}
                />
                <ProgressBar
                    who="TrialNum"
                    percent={bar2Percent}
                    description={bar2.toString()}
                    bgclass={status}
                    maxString={`MaxTrialNumber: ${trialProfile.MaxTrialNum}`}
                />
                <Row className="basic colorOfbasic mess">
                    <p>Best Default Metric</p>
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
                            <p>MaxDuration</p>
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