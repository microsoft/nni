import * as React from 'react';
import axios from 'axios';
import JSONTree from 'react-json-tree';
import { Row, Modal, Input, Table, Tabs } from 'antd';
const TabPane = Tabs.TabPane;
const { TextArea } = Input;
import { DOWNLOAD_IP } from '../../static/const';
import { TableObj } from '../../static/interface';
import { convertDuration } from '../../static/function';
import PaiTrialLog from '../logPath/PaiTrialLog';
import TrialLog from '../logPath/TrialLog';
import '../../static/style/tableStatus.css';
import '../../static/style/tableList.scss';

interface SuccessTableProps {
    tableSource: Array<TableObj>;
    trainingPlatform: string;
}

interface SuccessTableState {
    isShowLogModal: boolean;
    logContent: string;
}

class SuccessTable extends React.Component<SuccessTableProps, SuccessTableState> {

    public _isMounted = false;

    constructor(props: SuccessTableProps) {
        super(props);

        this.state = {
            isShowLogModal: false,
            logContent: ''
        };

    }

    showLogModalOverview = (id: string) => {
        axios(`${DOWNLOAD_IP}/trial_${id}.log`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    if (this._isMounted) {
                        this.setState(() => ({
                            logContent: res.data
                        }));
                    }
                }
            })
            .catch(error => {
                if (error.response.status === 500) {
                    if (this._isMounted) {
                        this.setState(() => ({
                            logContent: 'failed to get log message'
                        }));
                    }
                }
            });
        if (this._isMounted) {
            this.setState({
                isShowLogModal: true
            });
        }
    }

    hideLogModalOverview = () => {
        if (this._isMounted) {
            this.setState({
                isShowLogModal: false,
                logContent: '' // close modal, delete data
            });
        }
    }

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { tableSource, trainingPlatform } = this.props;
        const { isShowLogModal, logContent } = this.state;

        let bgColor = '';
        const columns = [{
            title: 'Trial No.',
            dataIndex: 'sequenceId',
            key: 'sequenceId',
            width: 140,
            className: 'tableHead'
        }, {
            title: 'Id',
            dataIndex: 'id',
            key: 'id',
            width: 60,
            className: 'tableHead idtitle',
            render: (text: string, record: TableObj) => {
                return (
                    <div>{record.id}</div>
                );
            },
        }, {
            title: 'Duration',
            dataIndex: 'duration',
            key: 'duration',
            width: 140,
            sorter: (a: TableObj, b: TableObj) => (a.duration as number) - (b.duration as number),
            render: (text: string, record: TableObj) => {
                let duration;
                if (record.duration) {
                    duration = convertDuration(record.duration);
                }
                return (
                    <div className="durationsty"><div>{duration}</div></div>
                );
            },
        }, {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            width: 150,
            className: 'tableStatus',
            render: (text: string, record: TableObj) => {
                bgColor = record.status;
                return (
                    <div className={`${bgColor} commonStyle`}>
                        {record.status}
                    </div>
                );
            }
        }, {
            title: 'Default Metric',
            dataIndex: 'acc',
            key: 'acc',
            sorter: (a: TableObj, b: TableObj) => (a.acc as number) - (b.acc as number),
            render: (text: string, record: TableObj) => {
                const accuracy = record.acc;
                let wei = 0;
                if (accuracy) {
                    if (accuracy.toString().indexOf('.') !== -1) {
                        wei = accuracy.toString().length - accuracy.toString().indexOf('.') - 1;
                    }
                }
                return (
                    <div>
                        {
                            record.acc
                                ?
                                wei > 6
                                    ?
                                    record.acc.toFixed(6)
                                    :
                                    record.acc
                                :
                                '--'
                        }
                    </div>
                );
            }
            // width: 150
        }];

        const openRow = (record: TableObj) => {
            let isHasParameters = true;
            if (record.description.parameters.error) {
                isHasParameters = false;
            }
            const openRowDataSource = {
                parameters: record.description.parameters
            };
            const logPathRow = record.description.logPath !== undefined
                ?
                record.description.logPath
                :
                'This trial\'s logPath are not available.';
            return (
                <pre id="description" className="hyperpar">
                    <Row className="openRowContent">
                        <Tabs tabPosition="left" className="card">
                            <TabPane tab="Parameters" key="1">
                                {
                                    isHasParameters
                                        ?
                                        <JSONTree
                                            hideRoot={true}
                                            shouldExpandNode={() => true}  // default expandNode
                                            getItemString={() => (<span />)}  // remove the {} items
                                            data={openRowDataSource}
                                        />
                                        :
                                        <div className="logpath">
                                            <span className="logName">Error: </span>
                                            <span className="error">'This trial's parameters are not available.'</span>
                                        </div>
                                }
                            </TabPane>
                            <TabPane tab="Log" key="2">
                                {
                                    trainingPlatform === 'pai' || trainingPlatform === 'kubeflow'
                                        ?
                                        <PaiTrialLog
                                            logStr={logPathRow}
                                            id={record.id}
                                            showLogModal={this.showLogModalOverview}
                                        />
                                        :
                                        <TrialLog logStr={logPathRow} id={record.id} />
                                }
                            </TabPane>
                        </Tabs>
                    </Row>
                </pre>
            );
        };
        return (
            <div className="tabScroll">
                <Table
                    columns={columns}
                    expandedRowRender={openRow}
                    dataSource={tableSource}
                    className="commonTableStyle"
                    pagination={false}
                />
                {/* trial log modal */}
                <Modal
                    title="trial log"
                    visible={isShowLogModal}
                    onCancel={this.hideLogModalOverview}
                    footer={null}
                    destroyOnClose={true}
                    width="80%"
                >
                    <div id="trialLogContent" style={{ height: window.innerHeight * 0.6 }}>
                        <TextArea
                            value={logContent}
                            disabled={true}
                            className="logcontent"
                        />
                    </div>
                </Modal>
            </div>
        );
    }
}

export default SuccessTable;