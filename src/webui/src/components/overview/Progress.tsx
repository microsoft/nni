import * as React from 'react';
import { Row, Col, Popover, message } from 'antd';
import axios from 'axios';
import { MANAGER_IP } from '../../static/const';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { convertTime } from '../../static/function';
import ConcurrencyInput from './NumInput';
import ProgressBar from './ProgressItem';
import LogDrawer from '../Modal/LogDrawer';
import '../../static/style/progress.scss';
import '../../static/style/probar.scss';

interface ProgressProps {
    concurrency: number;
    bestAccuracy: number;
    changeConcurrency: (val: number) => void;
    experimentUpdateBroadcast: number;
}

interface ProgressState {
    isShowLogDrawer: boolean;
}

class Progressed extends React.Component<ProgressProps, ProgressState> {
    constructor(props: ProgressProps) {
        super(props);
        this.state = {
            isShowLogDrawer: false
        };
    }

    editTrialConcurrency = async (userInput: string) => {
        if (!userInput.match(/^[1-9]\d*$/)) {
            message.error('Please enter a positive integer!', 2);
            return;
        }
        const newConcurrency = parseInt(userInput, 10);
        if (newConcurrency === this.props.concurrency) {
            message.info(`Trial concurrency has not changed`, 2);
            return;
        }

        const newProfile = Object.assign({}, EXPERIMENT.profile);
        newProfile.params.trialConcurrency = newConcurrency;

        // rest api, modify trial concurrency value
        try {
            const res = await axios.put(`${MANAGER_IP}/experiment`, newProfile, {
                params: { update_type: 'TRIAL_CONCURRENCY' }
            });
            if (res.status === 200) {
                message.success(`Successfully updated trial concurrency`);
                // NOTE: should we do this earlier in favor of poor networks?
                this.props.changeConcurrency(newConcurrency);
            }
        } catch (error) {
            if (error.response && error.response.data.error) {
                message.error(`Failed to update trial concurrency\n${error.response.data.error}`);
            } else if (error.response) {
                message.error(`Failed to update trial concurrency\nServer responsed ${error.response.status}`);
            } else if (error.message) {
                message.error(`Failed to update trial concurrency\n${error.message}`);
            } else {
                message.error(`Failed to update trial concurrency\nUnknown error`);
            }
        }
    }

    isShowDrawer = () => {
        this.setState({ isShowLogDrawer: true });
    }

    closeDrawer = () => {
        this.setState({ isShowLogDrawer: false });
    }

    render() {
        const { bestAccuracy } = this.props;
        const { isShowLogDrawer } = this.state;

        const count = TRIALS.countStatus();
        const stoppedCount = count.get('USER_CANCELED')! + count.get('SYS_CANCELED')! + count.get('EARLY_STOPPED')!;
        const bar2 = count.get('RUNNING')! + count.get('SUCCEEDED')! + count.get('FAILED')! + stoppedCount;

        const bar2Percent = (bar2 / EXPERIMENT.profile.params.maxTrialNum) * 100;
        const percent = (EXPERIMENT.profile.execDuration / EXPERIMENT.profile.params.maxExecDuration) * 100;
        const remaining = convertTime(EXPERIMENT.profile.params.maxExecDuration - EXPERIMENT.profile.execDuration);
        const maxDuration = convertTime(EXPERIMENT.profile.params.maxExecDuration);
        const maxTrialNum = EXPERIMENT.profile.params.maxTrialNum;
        const execDuration = convertTime(EXPERIMENT.profile.execDuration);

        let errorContent;
        if (EXPERIMENT.error) {
            errorContent = (
                <div className="errors">
                    {EXPERIMENT.error}
                    <div><a href="#" onClick={this.isShowDrawer}>Learn about</a></div>
                </div>
            );
        }
        return (
            <Row className="progress" id="barBack">
                <Row className="basic lineBasic">
                    <p>Status</p>
                    <div className="status">
                        <span className={EXPERIMENT.status}>{EXPERIMENT.status}</span>
                        {
                            EXPERIMENT.status === 'ERROR'
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
                    description={execDuration}
                    bgclass={EXPERIMENT.status}
                    maxString={`Max duration: ${maxDuration}`}
                />
                <ProgressBar
                    who="Trial numbers"
                    percent={bar2Percent}
                    description={bar2.toString()}
                    bgclass={EXPERIMENT.status}
                    maxString={`Max trial number: ${maxTrialNum}`}
                />
                <Row className="basic colorOfbasic mess">
                    <p>Best metric</p>
                    <div>{isNaN(bestAccuracy) ? 'N/A' : bestAccuracy.toFixed(6)}</div>
                </Row>
                <Row className="mess">
                    <Col span={6}>
                        <Row className="basic colorOfbasic">
                            <p>Spent</p>
                            <div>{execDuration}</div>
                        </Row>
                    </Col>
                    <Col span={6}>
                        <Row className="basic colorOfbasic">
                            <p>Remaining</p>
                            <div className="time">{remaining}</div>
                        </Row>
                    </Col>
                    <Col span={12}>
                        {/* modify concurrency */}
                        <p>Concurrency</p>
                        <ConcurrencyInput value={this.props.concurrency} updateValue={this.editTrialConcurrency} />
                    </Col>
                </Row>
                <Row className="mess">
                    <Col span={6}>
                        <Row className="basic colorOfbasic">
                            <p>Running</p>
                            <div>{count.get('RUNNING')}</div>
                        </Row>
                    </Col>
                    <Col span={6}>
                        <Row className="basic colorOfbasic">
                            <p>Succeeded</p>
                            <div>{count.get('SUCCEEDED')}</div>
                        </Row>
                    </Col>
                    <Col span={6}>
                        <Row className="basic">
                            <p>Stopped</p>
                            <div>{stoppedCount}</div>
                        </Row>
                    </Col>
                    <Col span={6}>
                        <Row className="basic">
                            <p>Failed</p>
                            <div>{count.get('FAILED')}</div>
                        </Row>
                    </Col>
                </Row>
                {/* learn about click -> default active key is dispatcher. */}
                {isShowLogDrawer ? (
                    <LogDrawer
                        closeDrawer={this.closeDrawer}
                        activeTab="dispatcher"
                    />
                ) : null}
            </Row>
        );
    }
}

export default Progressed;
