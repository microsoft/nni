import * as React from 'react';
import axios from 'axios';
import ReactEcharts from 'echarts-for-react';
import { Row, Table, Button, Popconfirm, Modal, Checkbox } from 'antd';
const CheckboxGroup = Checkbox.Group;
import { MANAGER_IP, trialJobStatus, COLUMN, COLUMN_INDEX } from '../../static/const';
import { convertDuration, intermediateGraphOption, killJob } from '../../static/function';
import { TableObj, TrialJob } from '../../static/interface';
import OpenRow from '../public-child/OpenRow';
import IntermediateVal from '../public-child/IntermediateVal'; // table default metric column
import '../../static/style/search.scss';
require('../../static/style/tableStatus.css');
require('../../static/style/logPath.scss');
require('../../static/style/search.scss');
require('../../static/style/table.scss');
require('../../static/style/button.scss');
require('../../static/style/openRow.scss');
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
    updateList: Function;
    platform: string;
    logCollection: boolean;
    isMultiPhase: boolean;
}

interface TableListState {
    intermediateOption: object;
    modalVisible: boolean;
    isObjFinal: boolean;
    isShowColumn: boolean;
    columnSelected: Array<string>; // user select columnKeys
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
            columnSelected: COLUMN
        };
    }

    showIntermediateModal = (id: string) => {

        axios(`${MANAGER_IP}/metric-data/${id}`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const intermediateArr: number[] = [];
                    // support intermediate result is dict
                    Object.keys(res.data).map(item => {
                        const temp = JSON.parse(res.data[item].data);
                        if (typeof temp === 'object') {
                            intermediateArr.push(temp.default);
                        } else {
                            intermediateArr.push(temp);
                        }
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
                case 'ID':
                case 'Duration':
                case 'Status':
                case 'Operation':
                case 'Default':
                case 'Intermediate result':
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
        const { platform, logCollection, isMultiPhase } = this.props;
        return (
            <OpenRow
                trainingPlatform={platform}
                record={record}
                logCollection={logCollection}
                multiphase={isMultiPhase}
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

        const { entries, tableSource, updateList } = this.props;
        const { intermediateOption, modalVisible, isShowColumn, columnSelected } = this.state;
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
                case 'ID':
                    showColumn.push({
                        title: 'ID',
                        dataIndex: 'id',
                        key: 'id',
                        width: 60,
                        className: 'tableHead leftTitle',
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
                            if (record.duration !== undefined) {
                                if (record.duration > 0 && record.duration < 1) {
                                    duration = `${record.duration}s`;
                                } else {
                                    duration = convertDuration(record.duration);
                                }
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
                        onFilter: (value: string, record: TableObj) => {
                            return record.status.indexOf(value) === 0;
                        },
                        // onFilter: (value: string, record: TableObj) => record.status.indexOf(value) === 0,
                        sorter: (a: TableObj, b: TableObj): number => a.status.localeCompare(b.status)
                    });
                    break;
                case 'Default':
                    showColumn.push({
                        title: 'Default metric',
                        className: 'leftTitle',
                        dataIndex: 'acc',
                        key: 'acc',
                        width: 120,
                        sorter: (a: TableObj, b: TableObj) => {
                            const aa = a.description.intermediate;
                            const bb = b.description.intermediate;
                            if (aa !== undefined && bb !== undefined) {
                                return aa[aa.length - 1] - bb[bb.length - 1];
                            } else {
                                return NaN;
                            }
                        },
                        render: (text: string, record: TableObj) => {
                            return (
                                <IntermediateVal record={record} />
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

                case 'Intermediate result':
                    showColumn.push({
                        title: 'Intermediate result',
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
                            const temp = record.acc;
                            let decimals = 0;
                            let other = '';
                            if (temp !== undefined) {
                                if (temp[item].toString().indexOf('.') !== -1) {
                                    decimals = temp[item].toString().length - temp[item].toString().indexOf('.') - 1;
                                    if (decimals > 6) {
                                        other = `${temp[item].toFixed(6)}`;
                                    } else {
                                        other = temp[item].toString();
                                    }
                                }
                            } else {
                                other = '--';
                            }
                            return (
                                <div>{other}</div>
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
                        dataSource={tableSource}
                        className="commonTableStyle"
                        pagination={{ pageSize: entries }}
                    />
                    {/* Intermediate Result Modal */}
                    <Modal
                        title="Intermediate result"
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
