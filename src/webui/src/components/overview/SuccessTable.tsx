import * as React from 'react';
import axios from 'axios';
import { Modal, Input, Table } from 'antd';
const { TextArea } = Input;
import OpenRow from '../public-child/OpenRow';
import DefaultMetric from '../public-child/DefaultMetrc';
import { DOWNLOAD_IP } from '../../static/const';
import { TableObj } from '../../static/interface';
import { convertDuration } from '../../static/function';
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

    openRow = (record: TableObj) => {
        const { trainingPlatform } = this.props;
        return (
            <OpenRow
                showLogModalOverview={this.showLogModalOverview}
                trainingPlatform={trainingPlatform}
                record={record}
            />
        );
    }

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { tableSource } = this.props;
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
            sorter: (a: TableObj, b: TableObj) => {
                if (a.acc !== undefined && b.acc !== undefined) {
                    return JSON.parse(a.acc.default) - JSON.parse(b.acc.default);
                } else {
                    return NaN;
                }
            },
            render: (text: string, record: TableObj) => {
                return (
                    <DefaultMetric record={record} />
                );
            }
        }];
        return (
            <div className="tabScroll" >
                <Table
                    columns={columns}
                    expandedRowRender={this.openRow}
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
            </div >
        );
    }
}
export default SuccessTable;