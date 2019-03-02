import * as React from 'react';
import axios from 'axios';
import ReactEcharts from 'echarts-for-react';
import {
    Row, Input, Table, Button, Popconfirm, Modal, Checkbox
} from 'antd';
const { TextArea } = Input;
const CheckboxGroup = Checkbox.Group;
import { MANAGER_IP, DOWNLOAD_IP, trialJobStatus, COLUMN, COLUMN_INDEX } from '../../static/const';
import { convertDuration, intermediateGraphOption, killJob } from '../../static/function';
import { TableObj, TrialJob } from '../../static/interface';
import OpenRow from '../public-child/OpenRow';
import DefaultMetric from '../public-child/DefaultMetrc';
import '../../static/style/search.scss';
require('../../static/style/tableStatus.css');
require('../../static/style/logPath.scss');
require('../../static/style/search.scss');
require('../../static/style/table.scss');
require('../../static/style/button.scss');
require('../../static/style/tableList.scss');
const echarts = require('echarts/lib/echarts');
require('echarts/lib/chart/line');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
echarts.registerTheme('my_theme', {
    color: '#3c8dbc'
});

interface TableListProps {
    entries: number;
    tableSource: Array<TableObj>;
    searchResult: Array<TableObj>;
    updateList: Function;
    isHasSearch: boolean;
    platform: string;
}

interface TableListState {
    intermediateOption: object;
    modalVisible: boolean;
    isObjFinal: boolean;
    isShowColumn: boolean;
    columnSelected: Array<string>; // user select columnKeys
    logModal: boolean;
    logMessage: string;
}

interface ColumnIndex {
    name: string;
    index: number;
}

class TableList extends React.Component<TableListProps, TableListState> {

    public _isMounted = false;
    public intervalTrialLog = 10;
    public _trialId: string;

    constructor(props: TableListProps) {
        super(props);

        this.state = {
            intermediateOption: {},
            modalVisible: false,
            isObjFinal: false,
            isShowColumn: false,
            logModal: false,
            columnSelected: COLUMN,
            logMessage: ''
        };
    }

    showIntermediateModal = (id: string) => {

        axios(`${MANAGER_IP}/metric-data/${id}`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const intermediateArr: number[] = [];
                    Object.keys(res.data).map(item => {
                        intermediateArr.push(parseFloat(res.data[item].data));
                    });
                    const intermediate = intermediateGraphOption(intermediateArr, id);
                    if (this._isMounted) {
                        this.setState(() => ({
                            intermediateOption: intermediate
                        }));
                    }
                }
            });
        if (this._isMounted) {
            this.setState({
                modalVisible: true
            });
        }
    }

    hideIntermediateModal = () => {
        if (this._isMounted) {
            this.setState({
                modalVisible: false
            });
        }
    }

    updateTrialLogMessage = (id: string) => {
        this._trialId = id;
        axios(`${DOWNLOAD_IP}/trial_${this._trialId}.log`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    if (this._isMounted) {
                        this.setState(() => ({
                            logMessage: res.data
                        }));
                    }
                }
            })
            .catch(error => {
                if (error.response.status === 500) {
                    if (this._isMounted) {
                        this.setState(() => ({
                            logMessage: 'failed to get log message'
                        }));
                    }
                }
            });
    }

    showLogModal = (id: string) => {
        this.updateTrialLogMessage(id);
        this.intervalTrialLog = window.setInterval(this.updateTrialLogMessage.bind(this, this._trialId), 10000);
        if (this._isMounted) {
            this.setState({
                logModal: true
            });
        }
    }

    hideLogModal = () => {
        window.clearInterval(this.intervalTrialLog);
        if (this._isMounted) {
            this.setState({
                logModal: false,
                logMessage: ''
            });
        }
    }

    hideShowColumnModal = () => {
        if (this._isMounted) {
            this.setState({
                isShowColumn: false
            });
        }
    }

    // click add column btn, just show the modal of addcolumn
    addColumn = () => {
        // show user select check button
        if (this._isMounted) {
            this.setState({
                isShowColumn: true
            });
        }
    }

    // checkbox for coloumn
    selectedColumn = (checkedValues: Array<string>) => {
        let count = 6;
        const want: Array<object> = [];
        const finalKeys: Array<string> = [];
        const wantResult: Array<string> = [];
        Object.keys(checkedValues).map(m => {
            switch (checkedValues[m]) {
                case 'Trial No.':
                case 'Id':
                case 'Duration':
                case 'Status':
                case 'Operation':
                case 'Default':
                case 'Intermediate Result':
                    break;
                default:
                    finalKeys.push(checkedValues[m]);
            }
        });

        Object.keys(finalKeys).map(n => {
            want.push({
                name: finalKeys[n],
                index: count++
            });
        });

        Object.keys(checkedValues).map(item => {
            const temp = checkedValues[item];
            Object.keys(COLUMN_INDEX).map(key => {
                const index = COLUMN_INDEX[key];
                if (index.name === temp) {
                    want.push(index);
                }
            });
        });

        want.sort((a: ColumnIndex, b: ColumnIndex) => {
            return a.index - b.index;
        });

        Object.keys(want).map(i => {
            wantResult.push(want[i].name);
        });

        if (this._isMounted) {
            this.setState(() => ({ columnSelected: wantResult }));
        }
    }

    openRow = (record: TableObj) => {
        const { platform } = this.props;
        return (
            <OpenRow
                showLogModalOverview={this.showLogModal}
                trainingPlatform={platform}
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

        const { entries, tableSource, searchResult, isHasSearch, updateList } = this.props;
        const { intermediateOption, modalVisible, isShowColumn, columnSelected,
            logMessage, logModal
        } = this.state;
        let showTitle = COLUMN;
        let bgColor = '';
        const trialJob: Array<TrialJob> = [];
        const showColumn: Array<object> = [];
        if (tableSource.length >= 1) {
            const temp = tableSource[0].acc;
            if (temp !== undefined && typeof temp === 'object') {
                if (this._isMounted) {
                    // concat default column and finalkeys
                    const item = Object.keys(temp);
                    const want: Array<string> = [];
                    Object.keys(item).map(key => {
                        if (item[key] !== 'default') {
                            want.push(item[key]);
                        }
                    });
                    showTitle = COLUMN.concat(want);
                }
            }
        }
        trialJobStatus.map(item => {
            trialJob.push({
                text: item,
                value: item
            });
        });
        Object.keys(columnSelected).map(key => {
            const item = columnSelected[key];
            switch (item) {
                case 'Trial No.':
                    showColumn.push({
                        title: 'Trial No.',
                        dataIndex: 'sequenceId',
                        key: 'sequenceId',
                        width: 120,
                        className: 'tableHead',
                        sorter:
                            (a: TableObj, b: TableObj) =>
                                (a.sequenceId as number) - (b.sequenceId as number)
                    });
                    break;
                case 'Id':
                    showColumn.push({
                        title: 'Id',
                        dataIndex: 'id',
                        key: 'id',
                        width: 60,
                        className: 'tableHead idtitle',
                        // the sort of string
                        sorter: (a: TableObj, b: TableObj): number => a.id.localeCompare(b.id),
                        render: (text: string, record: TableObj) => {
                            return (
                                <div>{record.id}</div>
                            );
                        }
                    });
                    break;
                case 'Duration':
                    showColumn.push({
                        title: 'Duration',
                        dataIndex: 'duration',
                        key: 'duration',
                        width: 140,
                        // the sort of number
                        sorter: (a: TableObj, b: TableObj) => (a.duration as number) - (b.duration as number),
                        render: (text: string, record: TableObj) => {
                            let duration;
                            if (record.duration !== undefined && record.duration > 0) {
                                duration = convertDuration(record.duration);
                            } else {
                                duration = 0;
                            }
                            return (
                                <div className="durationsty"><div>{duration}</div></div>
                            );
                        },
                    });
                    break;
                case 'Status':
                    showColumn.push({
                        title: 'Status',
                        dataIndex: 'status',
                        key: 'status',
                        width: 150,
                        className: 'tableStatus',
                        render: (text: string, record: TableObj) => {
                            bgColor = record.status;
                            return (
                                <span className={`${bgColor} commonStyle`}>{record.status}</span>
                            );
                        },
                        filters: trialJob,
                        onFilter: (value: string, record: TableObj) => record.status.indexOf(value) === 0,
                        sorter: (a: TableObj, b: TableObj): number => a.status.localeCompare(b.status)
                    });
                    break;
                case 'Default':
                    showColumn.push({
                        title: 'Default Metric',
                        dataIndex: 'acc',
                        key: 'acc',
                        width: 200,
                        sorter: (a: TableObj, b: TableObj) => {
                            if (a.acc !== undefined && b.acc !== undefined) {
                                return JSON.parse(a.acc.default) - JSON.parse(b.acc.default);
                            } else {
                                return NaN;
                            }
                        },
                        render: (text: string, record: TableObj) => {
                            return (
                                <DefaultMetric record={record}/>
                            );
                        }
                    });
                    break;
                case 'Operation':
                    showColumn.push({
                        title: 'Operation',
                        dataIndex: 'operation',
                        key: 'operation',
                        width: 90,
                        render: (text: string, record: TableObj) => {
                            let trialStatus = record.status;
                            let flagKill = false;
                            if (trialStatus === 'RUNNING') {
                                flagKill = true;
                            } else {
                                flagKill = false;
                            }
                            return (
                                flagKill
                                    ?
                                    (
                                        <Popconfirm
                                            title="Are you sure to cancel this trial?"
                                            onConfirm={killJob.
                                                bind(this, record.key, record.id, record.status, updateList)}
                                        >
                                            <Button type="primary" className="tableButton">Kill</Button>
                                        </Popconfirm>
                                    )
                                    :
                                    (
                                        <Button
                                            type="primary"
                                            className="tableButton"
                                            disabled={true}
                                        >
                                            Kill
                                        </Button>
                                    )
                            );
                        },
                    });
                    break;

                case 'Intermediate Result':
                    showColumn.push({
                        title: 'Intermediate Result',
                        dataIndex: 'intermediate',
                        key: 'intermediate',
                        width: '16%',
                        render: (text: string, record: TableObj) => {
                            return (
                                <Button
                                    type="primary"
                                    className="tableButton"
                                    onClick={this.showIntermediateModal.bind(this, record.id)}
                                >
                                    Intermediate
                                </Button>
                            );
                        },
                    });
                    break;
                default:
                    showColumn.push({
                        title: item,
                        dataIndex: item,
                        key: item,
                        width: 150,
                        render: (text: string, record: TableObj) => {
                            return (
                                <div>
                                    {
                                        record.acc
                                            ?
                                            record.acc[item]
                                            :
                                            '--'
                                    }
                                </div>
                            );
                        }
                    });
            }
        });

        return (
            <Row className="tableList">
                <div id="tableList">
                    <Table
                        columns={showColumn}
                        expandedRowRender={this.openRow}
                        dataSource={isHasSearch ? searchResult : tableSource}
                        className="commonTableStyle"
                        pagination={{ pageSize: entries }}
                    />
                    {/* Intermediate Result Modal */}
                    <Modal
                        title="Intermediate Result"
                        visible={modalVisible}
                        onCancel={this.hideIntermediateModal}
                        footer={null}
                        destroyOnClose={true}
                        width="80%"
                    >
                        <ReactEcharts
                            option={intermediateOption}
                            style={{
                                width: '100%',
                                height: 0.7 * window.innerHeight
                            }}
                            theme="my_theme"
                        />
                    </Modal>

                    {/* trial log modal */}
                    <Modal
                        title="trial log"
                        visible={logModal}
                        onCancel={this.hideLogModal}
                        footer={null}
                        destroyOnClose={true}
                        width="80%"
                    >
                        <div id="trialLogContent" style={{ height: window.innerHeight * 0.6 }}>
                            <TextArea
                                value={logMessage}
                                disabled={true}
                                className="logcontent"
                            />
                        </div>
                    </Modal>
                </div>
                {/* Add Column Modal */}
                <Modal
                    title="Table Title"
                    visible={isShowColumn}
                    onCancel={this.hideShowColumnModal}
                    footer={null}
                    destroyOnClose={true}
                    width="40%"
                >
                    <CheckboxGroup
                        options={showTitle}
                        defaultValue={columnSelected}
                        onChange={this.selectedColumn}
                        className="titleColumn"
                    />
                </Modal>

            </Row>
        );
    }
}

export default TableList;
