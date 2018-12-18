import * as React from 'react';
import {
    Row, Col,
    Tooltip, Button
} from 'antd';
import axios from 'axios';
import { DOWNLOAD_IP } from '../../static/const';
import { Experiment } from '../../static/interface';

interface BasicInfoProps {
    trialProfile: Experiment;
    status: string;
}

interface BasicInfoState {
    disablednnilog: boolean;
    disabledispatch: boolean;
}

class BasicInfo extends React.Component<BasicInfoProps, BasicInfoState> {
    public _isMounted = false;
    constructor(props: BasicInfoProps) {
        super(props);
        this.state = {
            disablednnilog: false,
            disabledispatch: false
        };
    }

    downnnimanagerLog = () => {
        if (this._isMounted) { this.setState({ disablednnilog: true }); }
        axios(`${DOWNLOAD_IP}/nnimanager.log`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const nniLogfile = res.data;
                    const aTag = document.createElement('a');
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
                    const file = new Blob([nniLogfile], { type: 'application/json' });
                    aTag.download = 'nnimanagerLog.json';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'nnimanagerLog.json';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                    if (this._isMounted) { this.setState({ disablednnilog: false }); }
                }
            });
    }

    downDispatcherlog = () => {
        if (this._isMounted) { this.setState({ disabledispatch: true }); }
        axios(`${DOWNLOAD_IP}/dispatcher.log`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const dispatchLogfile = res.data;
                    const aTag = document.createElement('a');
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
                    const file = new Blob([dispatchLogfile], { type: 'application/json' });
                    aTag.download = 'dispatcherLog.json';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'dispatcherLog.json';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                    if (this._isMounted) { this.setState({ disabledispatch: false }); }
                }
            });
    }

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }
    
    render() {
        const { trialProfile } = this.props;
        const {disablednnilog, disabledispatch} = this.state;
        return (
            <Row className="main">
                <Col span={8} className="padItem basic">
                    <p>Name</p>
                    <div>{trialProfile.experName}</div>
                    <p>ID</p>
                    <div>{trialProfile.id}</div>
                </Col>
                <Col span={8} className="padItem basic">
                    <p>Start Time</p>
                    <div className="nowrap">
                        {new Date(trialProfile.startTime).toLocaleString('en-US')}
                    </div>
                    <p>End Time</p>
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
                    <p>
                        <span>LogPath</span>
                        <Button
                            type="primary"
                            className="tableButton downStyle downMargin"
                            onClick={this.downnnimanagerLog}
                            disabled={disablednnilog}
                        >
                            <img
                                className="downloadLog"
                                src={require('../../static/img/icon/download.png')}
                                alt="download nnimanager log file"
                                title="download nnimanager log file"
                            />
                        </Button>
                        <Button
                            type="primary"
                            className="tableButton downStyle downMargin"
                            onClick={this.downDispatcherlog}
                            disabled={disabledispatch}
                        >
                            <img
                                className="downloadLog"
                                src={require('../../static/img/icon/download.png')}
                                alt="download dispatcher log file"
                                title="download dispatcher log file"
                            />
                        </Button>
                    </p>
                    <div className="nowrap">
                        <Tooltip placement="top" title={trialProfile.logDir}>
                            {trialProfile.logDir}
                        </Tooltip>
                    </div>
                    <p>TrainingPlatform</p>
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