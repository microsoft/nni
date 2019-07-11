import * as React from 'react';
import axios from 'axios';
import { Drawer, Tabs, Row, Col, Button, Icon } from 'antd';
import { DOWNLOAD_IP } from '../../static/const';
import { downFile } from '../../static/function';
const { TabPane } = Tabs;
import MonacoHTML from '../public-child/MonacoEditor';
import '../../static/style/logDrawer.scss';

interface LogDrawerProps {
    isVisble: boolean;
    closeDrawer: () => void;
    activeTab?: string;
}

interface LogDrawerState {
    nniManagerLogStr: string;
    dispatcherLogStr: string;
    isLoading: boolean;
    isLoadispatcher: boolean;
}

class LogDrawer extends React.Component<LogDrawerProps, LogDrawerState> {

    public _isLogDrawer: boolean;
    constructor(props: LogDrawerProps) {
        super(props);

        this.state = {
            nniManagerLogStr: 'nnimanager',
            dispatcherLogStr: 'dispatcher',
            isLoading: false,
            isLoadispatcher: false
        };
    }

    getNNImanagerLogmessage = () => {
        if (this._isLogDrawer === true) {
            this.setState({ isLoading: true }, () => {
                axios(`${DOWNLOAD_IP}/nnimanager.log`, {
                    method: 'GET'
                })
                    .then(res => {
                        if (res.status === 200) {
                            setTimeout(() => { this.setNNImanager(res.data); }, 300);
                        }
                    });
            });
        }
    }

    setDispatcher = (value: string) => {
        if (this._isLogDrawer === true) {
            this.setState(() => ({ isLoadispatcher: false, dispatcherLogStr: value }));
        }
    }

    setNNImanager = (val: string) => {
        if (this._isLogDrawer === true) {
            this.setState(() => ({ isLoading: false, nniManagerLogStr: val }));
        }
    }

    getdispatcherLogmessage = () => {
        if (this._isLogDrawer === true) {
            this.setState({ isLoadispatcher: true }, () => {
                axios(`${DOWNLOAD_IP}/dispatcher.log`, {
                    method: 'GET'
                })
                    .then(res => {
                        if (res.status === 200) {
                            setTimeout(() => { this.setDispatcher(res.data); }, 300);
                        }
                    });
            });
        }
    }

    downloadNNImanager = () => {
        const { nniManagerLogStr } = this.state;
        downFile(nniManagerLogStr, 'nnimanager.log');
    }

    downloadDispatcher = () => {
        const { dispatcherLogStr } = this.state;
        downFile(dispatcherLogStr, 'dispatcher.log');
    }

    dispatcherHTML = () => {
        return (
            <div>
                <span>Dispatcher Log</span>
                <span className="refresh" onClick={this.getdispatcherLogmessage}>
                    <Icon type="sync" />
                </span>
            </div>
        );
    }

    nnimanagerHTML = () => {
        return (
            <div>
                <span>NNImanager Log</span>
                <span className="refresh" onClick={this.getNNImanagerLogmessage}><Icon type="sync" /></span>
            </div>
        );
    }

    componentDidMount() {
        this._isLogDrawer = true;
        this.getNNImanagerLogmessage();
        this.getdispatcherLogmessage();
    }

    componentWillReceiveProps(nextProps: LogDrawerProps) {
        const { isVisble, activeTab } = nextProps;
        if (isVisble === true) {
            if (activeTab === 'nnimanager') {
                this.getNNImanagerLogmessage();
            }
            if (activeTab === 'dispatcher') {
                this.getdispatcherLogmessage();
            }
        }
    }

    componentWillUnmount() {
        this._isLogDrawer = false;
    }

    render() {
        const { isVisble, closeDrawer, activeTab } = this.props;
        const { nniManagerLogStr, dispatcherLogStr, isLoadispatcher, isLoading } = this.state;
        const heights: number = window.innerHeight - 48; // padding top and bottom
        return (
            <Row>
                <Drawer
                    placement="right"
                    closable={false}
                    destroyOnClose={true}
                    onClose={closeDrawer}
                    visible={isVisble}
                    width="76%"
                    height={heights}
                    // className="logDrawer"
                >
                    <div className="card-container log-tab-body" style={{ height: heights }}>
                        <Tabs type="card" defaultActiveKey={activeTab}>
                            {/* <Tabs type="card" onTabClick={this.selectwhichLog} defaultActiveKey={activeTab}> */}
                            {/* <TabPane tab="Dispatcher Log" key="dispatcher"> */}
                            <TabPane tab={this.dispatcherHTML()} key="dispatcher">
                                <div>
                                    <MonacoHTML content={dispatcherLogStr} loading={isLoadispatcher} />
                                </div>
                                <Row className="buttons">
                                    <Col span={12}>
                                        <Button
                                            type="primary"
                                            onClick={this.downloadDispatcher}
                                        >
                                            Download
                                        </Button>
                                    </Col>
                                    <Col span={12} className="close">
                                        <Button
                                            type="default"
                                            onClick={closeDrawer}
                                        >
                                            Close
                                        </Button>
                                    </Col>
                                </Row>
                            </TabPane>
                            <TabPane tab={this.nnimanagerHTML()} key="nnimanager">
                                {/* <TabPane tab="NNImanager Log" key="nnimanager"> */}
                                <div>
                                    <MonacoHTML content={nniManagerLogStr} loading={isLoading} />
                                </div>
                                <Row className="buttons">
                                    <Col span={12} className="download">
                                        <Button
                                            type="primary"
                                            onClick={this.downloadNNImanager}
                                        >
                                            Download
                                        </Button>
                                    </Col>
                                    <Col span={12} className="close">
                                        <Button
                                            type="default"
                                            onClick={closeDrawer}
                                        >
                                            Close
                                        </Button>
                                    </Col>
                                </Row>
                            </TabPane>
                        </Tabs>
                    </div>
                </Drawer>
            </Row>
        );
    }
}

export default LogDrawer;
