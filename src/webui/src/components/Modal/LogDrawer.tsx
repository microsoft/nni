import * as React from 'react';
import axios from 'axios';
import { Drawer, Tabs, Row, Col, Button, Icon } from 'antd';
import { DOWNLOAD_IP } from '../../static/const';
import { downFile } from '../../static/function';
const { TabPane } = Tabs;
import MonacoHTML from '../public-child/MonacoEditor';
import '../../static/style/logDrawer.scss';

interface LogDrawerProps {
    closeDrawer: () => void;
    activeTab?: string;
}

interface LogDrawerState {
    nniManagerLogStr: string | null;
    dispatcherLogStr: string | null;
    isLoading: boolean;
    logDrawerHeight: number;
}

class LogDrawer extends React.Component<LogDrawerProps, LogDrawerState> {
    private timerId: number | undefined;

    constructor(props: LogDrawerProps) {
        super(props);

        this.state = {
            nniManagerLogStr: null,
            dispatcherLogStr: null,
            isLoading: true,
            logDrawerHeight: window.innerHeight - 48
        };
    }

    downloadNNImanager = () => {
        if (this.state.nniManagerLogStr !== null) {
            downFile(this.state.nniManagerLogStr, 'nnimanager.log');
        }
    }

    downloadDispatcher = () => {
        if (this.state.dispatcherLogStr !== null) {
            downFile(this.state.dispatcherLogStr, 'dispatcher.log');
        }
    }

    dispatcherHTML = () => {
        return (
            <div>
                <span>Dispatcher Log</span>
                <span className="refresh" onClick={this.manualRefresh}>
                    <Icon type="sync" />
                </span>
            </div>
        );
    }

    nnimanagerHTML = () => {
        return (
            <div>
                <span>NNImanager Log</span>
                <span className="refresh" onClick={this.manualRefresh}><Icon type="sync" /></span>
            </div>
        );
    }

    setLogDrawerHeight = () => {
        this.setState(() => ({ logDrawerHeight: window.innerHeight - 48 }));
    }

    async componentDidMount() {
        this.refresh();
        window.addEventListener('resize', this.setLogDrawerHeight);
    }

    componentWillUnmount() {
        window.clearTimeout(this.timerId);
        window.removeEventListener('resize', this.setLogDrawerHeight);
    }

    render() {
        const { closeDrawer, activeTab } = this.props;
        const { nniManagerLogStr, dispatcherLogStr, isLoading, logDrawerHeight } = this.state;
        return (
            <Row>
                <Drawer
                    placement="right"
                    closable={false}
                    destroyOnClose={true}
                    onClose={closeDrawer}
                    visible={true}
                    width="76%"
                    height={logDrawerHeight}
                // className="logDrawer"
                >
                    <div className="card-container log-tab-body">
                        <Tabs
                            type="card"
                            defaultActiveKey={activeTab}
                            style={{ height: logDrawerHeight, minHeight: 190 }}
                        >
                            {/* <Tabs type="card" onTabClick={this.selectwhichLog} defaultActiveKey={activeTab}> */}
                            {/* <TabPane tab="Dispatcher Log" key="dispatcher"> */}
                            <TabPane tab={this.dispatcherHTML()} key="dispatcher">
                                <div>
                                    <MonacoHTML
                                        content={dispatcherLogStr || 'Loading...'}
                                        loading={isLoading}
                                        height={logDrawerHeight - 104}
                                    />
                                </div>
                                <Row className="buttons">
                                    <Col span={12} className="download">
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
                                    <MonacoHTML
                                        content={nniManagerLogStr || 'Loading...'}
                                        loading={isLoading}
                                        height={logDrawerHeight - 104}
                                    />
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

    private refresh = () => {
        window.clearTimeout(this.timerId);
        const dispatcherPromise = axios.get(`${DOWNLOAD_IP}/dispatcher.log`);
        const nniManagerPromise = axios.get(`${DOWNLOAD_IP}/nnimanager.log`);
        dispatcherPromise.then(res => {
            if (res.status === 200) {
                this.setState({ dispatcherLogStr: res.data });
            }
        });
        nniManagerPromise.then(res => {
            if (res.status === 200) {
                this.setState({ nniManagerLogStr: res.data });
            }
        });
        Promise.all([dispatcherPromise, nniManagerPromise]).then(() => {
            this.setState({ isLoading: false });
            this.timerId = window.setTimeout(this.refresh, 300);
        });
    }

    private manualRefresh = () => {
        this.setState({ isLoading: true });
        this.refresh();
    }
}

export default LogDrawer;
