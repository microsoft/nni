import * as React from 'react';
import axios from 'axios';
import JSONTree from 'react-json-tree';
import ReactEcharts from 'echarts-for-react';
import {
    Row, Input, Table, Tabs, Button, Popconfirm, Modal, message, Checkbox
} from 'antd';
const { TextArea } = Input;
const TabPane = Tabs.TabPane;
const CheckboxGroup = Checkbox.Group;
import { MANAGER_IP, DOWNLOAD_IP, trialJobStatus, COLUMN, COLUMN_INDEX } from '../../static/const';
import { convertDuration } from '../../static/function';
import { TableObjFianl, TrialJob } from '../../static/interface';
import PaiTrialLog from '../logPath/PaiTrialLog';
import TrialLog from '../logPath/TrialLog';
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
    tableSource: Array<TableObjFianl>;
    searchResult: Array<TableObjFianl>;
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
                    const intermediate = this.intermediateGraphOption(intermediateArr, id);
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

    intermediateGraphOption = (intermediateArr: number[], id: string) => {
        const sequence: number[] = [];
        const lengthInter = intermediateArr.length;
        for (let i = 1; i <= lengthInter; i++) {
            sequence.push(i);
        }
        return {
            title: {
                text: id,
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    color: '#333',
                }
            },
            tooltip: {
                trigger: 'item'
            },
            xAxis: {
                name: 'Trial',
                data: sequence
            },
            yAxis: {
                name: 'Default Metric',
                type: 'value',
                data: intermediateArr
            },
            series: [{
                symbolSize: 6,
                type: 'scatter',
                data: intermediateArr
            }]
        };
    }

    // kill job
    killJob = (key: number, id: string, status: string) => {
        const { updateList } = this.props;
        axios(`${MANAGER_IP}/trial-jobs/${id}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8'
            }
        })
            .then(res => {
                if (res.status === 200) {
                    message.success('Cancel the job successfully');
                    // render the table
                    updateList();
                } else {
                    message.error('fail to cancel the job');
                }
            })
            .catch(error => {
                if (error.response.status === 500) {
                    if (error.response.data.error) {
                        message.error(error.response.data.error);
                    } else {
                        message.error('500 error, fail to cancel the job');
                    }
                }
            });
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
                case 'Trial No':
                case 'id':
                case 'duration':
                case 'status':
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

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {

        const { entries, tableSource, searchResult, isHasSearch, platform } = this.props;
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
                case 'Trial No':
                    showColumn.push({
                        title: 'Trial No.',
                        dataIndex: 'sequenceId',
                        key: 'sequenceId',
                        width: 120,
                        className: 'tableHead',
                        sorter:
                            (a: TableObjFianl, b: TableObjFianl) =>
                                (a.sequenceId as number) - (b.sequenceId as number)
                    });
                    break;
                case 'id':
                    showColumn.push({
                        title: 'Id',
                        dataIndex: 'id',
                        key: 'id',
                        width: 60,
                        className: 'tableHead idtitle',
                        // the sort of string
                        sorter: (a: TableObjFianl, b: TableObjFianl): number => a.id.localeCompare(b.id),
                        render: (text: string, record: TableObjFianl) => {
                            return (
                                <div>{record.id}</div>
                            );
                        }
                    });
                    break;
                case 'duration':
                    showColumn.push({
                        title: 'Duration',
                        dataIndex: 'duration',
                        key: 'duration',
                        width: 140,
                        // the sort of number
                        sorter: (a: TableObjFianl, b: TableObjFianl) => (a.duration as number) - (b.duration as number),
                        render: (text: string, record: TableObjFianl) => {
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
                case 'status':
                    showColumn.push({
                        title: 'Status',
                        dataIndex: 'status',
                        key: 'status',
                        width: 150,
                        className: 'tableStatus',
                        render: (text: string, record: TableObjFianl) => {
                            bgColor = record.status;
                            return (
                                <span className={`${bgColor} commonStyle`}>{record.status}</span>
                            );
                        },
                        filters: trialJob,
                        onFilter: (value: string, record: TableObjFianl) => record.status.indexOf(value) === 0,
                        sorter: (a: TableObjFianl, b: TableObjFianl): number => a.status.localeCompare(b.status)
                    });
                    break;
                case 'Default':
                    showColumn.push({
                        title: 'Default Metric',
                        dataIndex: 'acc',
                        key: 'acc',
                        width: 200,
                        sorter: (a: TableObjFianl, b: TableObjFianl) => {
                            if (a.acc !== undefined && b.acc !== undefined) {
                                return JSON.parse(a.acc.default) - JSON.parse(b.acc.default);
                            } else {
                                return NaN;
                            }
                        },
                        render: (text: string, record: TableObjFianl) => {
                            let accuracy;
                            if (record.acc !== undefined) {
                                accuracy = record.acc.default;
                            }
                            let wei = 0;
                            if (accuracy) {
                                if (accuracy.toString().indexOf('.') !== -1) {
                                    wei = accuracy.toString().length - accuracy.toString().indexOf('.') - 1;
                                }
                            }
                            return (
                                <div>
                                    {
                                        record.acc && record.acc.default
                                            ?
                                            wei > 6
                                                ?
                                                JSON.parse(record.acc.default).toFixed(6)
                                                :
                                                record.acc.default
                                            :
                                            '--'
                                    }
                                </div>
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
                        render: (text: string, record: TableObjFianl) => {
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
                                            onConfirm={this.killJob.bind(this, record.key, record.id, record.status)}
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
                        render: (text: string, record: TableObjFianl) => {
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
                        render: (text: string, record: TableObjFianl) => {
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

        const openRow = (record: TableObjFianl) => {
            let isHasParameters = true;
            if (record.description.parameters.error) {
                isHasParameters = false;
            }
            const parametersRow = {
                parameters: record.description.parameters
            };
            const logPathRow = record.description.logPath !== undefined
                ?
                record.description.logPath
                :
                'This trial\'s logPath are not available.';
            const isdisLogbutton = record.status === 'WAITING'
                ?
                true
                :
                false;
            return (
                <pre id="allList" className="hyperpar">
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
                                            data={parametersRow}
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
                                    platform === 'pai' || platform === 'kubeflow'
                                        ?
                                        <PaiTrialLog
                                            logStr={logPathRow}
                                            id={record.id}
                                            showLogModal={this.showLogModal}
                                            trialStatus={record.status}
                                            isdisLogbutton={isdisLogbutton}
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
            <Row className="tableList">
                <div id="tableList">
                    <Table
                        columns={showColumn}
                        expandedRowRender={openRow}
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
