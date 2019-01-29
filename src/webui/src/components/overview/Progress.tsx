import * as React from 'react';
import {
    Row, Col, Popover, Button, message
} from 'antd';
import axios from 'axios';
import { MANAGER_IP, CONTROLTYPE } from '../../static/const';
import { Experiment, TrialNumber } from '../../static/interface';
import { convertTime } from '../../static/function';
import ProgressBar from './ProgressItem';
import '../../static/style/progress.scss';
import '../../static/style/probar.scss';

interface ProgressProps {
    trialProfile: Experiment;
    trialNumber: TrialNumber;
    bestAccuracy: number;
    status: string;
    errors: string;
    updateFile: Function;
}

interface ProgressState {
    btnName: string;
    isEnable: boolean;
    userInputVal: string; // get user input
    cancelSty: string;
}

class Progressed extends React.Component<ProgressProps, ProgressState> {

    public conInput: HTMLInputElement | null;
    public _isMounted = false;
    constructor(props: ProgressProps) {
        super(props);
        this.state = {
            btnName: 'Edit',
            isEnable: true,
            userInputVal: this.props.trialProfile.runConcurren.toString(),
            cancelSty: 'none'
        };
    }

    editTrialConcurrency = () => {
        const { btnName } = this.state;
        if (this._isMounted) {
            if (btnName === 'Edit') {
                this.setState(() => ({
                    isEnable: false,
                    btnName: 'Save',
                    cancelSty: 'inline-block'
                }));
            } else {
                axios(`${MANAGER_IP}/experiment`, {
                    method: 'GET'
                })
                    .then(rese => {
                        if (rese.status === 200) {
                            const { userInputVal } = this.state;
                            const experimentFile = rese.data;
                            const trialConcurrency = experimentFile.params.trialConcurrency;
                            if (userInputVal !== undefined) {
                                if (userInputVal === trialConcurrency.toString() || userInputVal === '0') {
                                    message.info(
                                        `trialConcurrency's value is ${trialConcurrency}, you did not modify it`, 2);
                                } else {
                                    experimentFile.params.trialConcurrency = parseInt(userInputVal, 10);
                                    // rest api, modify trial concurrency value
                                    axios(`${MANAGER_IP}/experiment`, {
                                        method: 'PUT',
                                        headers: {
                                            'Content-Type': 'application/json;charset=utf-8'
                                        },
                                        data: experimentFile,
                                        params: {
                                            update_type: CONTROLTYPE[1]
                                        }
                                    }).then(res => {
                                        if (res.status === 200) {
                                            message.success(`Update ${CONTROLTYPE[1].toLocaleLowerCase()} 
                                            successfully`);
                                            // rerender trial profile message
                                            const { updateFile } = this.props;
                                            updateFile();
                                        }
                                    })
                                    .catch(error => {
                                        if (error.response.status === 500) {
                                            if (error.response.data.error) {
                                                message.error(error.response.data.error);
                                            } else {
                                                message.error(
                                                    `Update ${CONTROLTYPE[1].toLocaleLowerCase()} failed`);
                                            }
                                        }
                                    });
                                    // btn -> edit
                                    this.setState(() => ({
                                        btnName: 'Edit',
                                        isEnable: true,
                                        cancelSty: 'none'
                                    }));
                                }
                            }
                        }
                    });
            }
        }
    }

    cancelFunction = () => {
        const { trialProfile } = this.props;
        if (this._isMounted) {
            this.setState(
                () => ({
                    btnName: 'Edit',
                    isEnable: true,
                    cancelSty: 'none',
                }));
        }
        if (this.conInput !== null) {
            this.conInput.value = trialProfile.runConcurren.toString();
        }
    }

    getUserTrialConcurrency = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value.match(/^[1-9]\d*$/) || value === '') {
            this.setState(() => ({
                userInputVal: value
            }));
        } else {
            message.error('Please enter a positive integer!', 2);
            if (this.conInput !== null) {
                const { trialProfile } = this.props;
                this.conInput.value = trialProfile.runConcurren.toString();
            }
        }
    }

    componentWillReceiveProps() {
        const { trialProfile } = this.props;
        if (this.conInput !== null) {
            this.conInput.value = trialProfile.runConcurren.toString();
        }
    }

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { trialProfile, trialNumber, bestAccuracy, status, errors } = this.props;
        const { isEnable, btnName, cancelSty } = this.state;
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
                <Row className="basic lineBasic">
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
                    <Col span={10}>
                        <p>best metric</p>
                        <div>{bestAccuracy.toFixed(6)}</div>
                    </Col>
                    <Col span={14}>
                        {/* modify concurrency */}
                        <p>concurrency</p>
                        <Row className="inputBox">
                            <input
                                type="number"
                                disabled={isEnable}
                                onChange={this.getUserTrialConcurrency}
                                className="concurrencyInput"
                                ref={(input) => this.conInput = input}
                            />
                            <Button
                                type="primary"
                                className="tableButton editStyle"
                                onClick={this.editTrialConcurrency}
                            >{btnName}
                            </Button>
                            <Button
                                type="primary"
                                onClick={this.cancelFunction}
                                style={{ display: cancelSty, marginLeft: 1 }}
                                className="tableButton editStyle"
                            >
                                Cancel
                            </Button>
                        </Row>
                    </Col>
                </Row>
                <Row className="mess">
                    <Col span={8}>
                        <Row className="basic colorOfbasic">
                            <p>spent</p>
                            <div>{convertTime(trialProfile.execDuration)}</div>
                        </Row>
                    </Col>
                    <Col span={9}>
                        <Row className="basic colorOfbasic">
                            <p>remaining</p>
                            <div>{remaining}</div>
                        </Row>
                    </Col>
                    <Col span={7}>
                        <Row className="basic colorOfbasic">
                            <p>running</p>
                            <div>{trialNumber.runTrial}</div>
                        </Row>
                    </Col>
                </Row>
                <Row className="mess">
                    <Col span={8}>
                        <Row className="basic colorOfbasic">
                            <p>succeed</p>
                            <div>{trialNumber.succTrial}</div>
                        </Row>
                    </Col>
                    <Col span={9}>
                        <Row className="basic">
                            <p>stopped</p>
                            <div>{trialNumber.stopTrial}</div>
                        </Row>
                    </Col>
                    <Col span={7}>
                        <Row className="basic">
                            <p>failed</p>
                            <div>{trialNumber.failTrial}</div>
                        </Row>
                    </Col>
                </Row>
            </Row>
        );
    }
}

export default Progressed;