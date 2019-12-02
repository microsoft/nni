import * as React from 'react';
import axios from 'axios';
import ReactEcharts from 'echarts-for-react';
import { Row, Table, Button, Popconfirm, Modal, Checkbox, Select, Icon } from 'antd';
import { ColumnProps } from 'antd/lib/table';
const Option = Select.Option;
const CheckboxGroup = Checkbox.Group;
import { MANAGER_IP, trialJobStatus, COLUMN_INDEX, COLUMNPro } from '../../static/const';
import { convertDuration, formatTimestamp, intermediateGraphOption, killJob } from '../../static/function';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { TableRecord } from '../../static/interface';
import OpenRow from '../public-child/OpenRow';
import Compare from '../Modal/Compare';
import Customize from '../Modal/CustomizedTrial';
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
    pageSize: number;
    tableSource: Array<TableRecord>;
    columnList: Array<string>; // user select columnKeys
    changeColumn: (val: Array<string>) => void;
    trialsUpdateBroadcast: number;
}

interface TableListState {
    intermediateOption: object;
    modalVisible: boolean;
    isObjFinal: boolean;
    isShowColumn: boolean;
    selectRows: Array<TableRecord>;
    isShowCompareModal: boolean;
    selectedRowKeys: string[] | number[];
    intermediateData: Array<object>; // a trial's intermediate results (include dict)
    intermediateId: string;
    intermediateOtherKeys: Array<string>;
    isShowCustomizedModal: boolean;
    copyTrialId: string; // user copy trial to submit a new customized trial
}

interface ColumnIndex {
    name: string;
    index: number;
}

class TableList extends React.Component<TableListProps, TableListState> {

    public intervalTrialLog = 10;
    public _trialId: string;
    public tables: Table<TableRecord> | null;

    constructor(props: TableListProps) {
        super(props);

        this.state = {
            intermediateOption: {},
            modalVisible: false,
            isObjFinal: false,
            isShowColumn: false,
            isShowCompareModal: false,
            selectRows: [],
            selectedRowKeys: [], // close selected trial message after modal closed
            intermediateData: [],
            intermediateId: '',
            intermediateOtherKeys: [],
            isShowCustomizedModal: false,
            copyTrialId: ''
        };
    }

    showIntermediateModal = async (id: string) => {
        const res = await axios.get(`${MANAGER_IP}/metric-data/${id}`);
        if (res.status === 200) {
            const intermediateArr: number[] = [];
            // support intermediate result is dict because the last intermediate result is
            // final result in a succeed trial, it may be a dict.
            // get intermediate result dict keys array
            let otherkeys: Array<string> = ['default'];
            if (res.data.length !== 0) {
                otherkeys = Object.keys(JSON.parse(res.data[0].data));
            }
            // intermediateArr just store default val
            Object.keys(res.data).map(item => {
                const temp = JSON.parse(res.data[item].data);
                if (typeof temp === 'object') {
                    intermediateArr.push(temp.default);
                } else {
                    intermediateArr.push(temp);
                }
            });
            const intermediate = intermediateGraphOption(intermediateArr, id);
            this.setState({
                intermediateData: res.data, // store origin intermediate data for a trial
                intermediateOption: intermediate,
                intermediateOtherKeys: otherkeys,
                intermediateId: id
            });
        }
        this.setState({ modalVisible: true });
    }

    // intermediate button click -> intermediate graph for each trial
    // support intermediate is dict
    selectOtherKeys = (value: string) => {

        const isShowDefault: boolean = value === 'default' ? true : false;
        const { intermediateData, intermediateId } = this.state;
        const intermediateArr: number[] = [];
        // just watch default key-val
        if (isShowDefault === true) {
            Object.keys(intermediateData).map(item => {
                const temp = JSON.parse(intermediateData[item].data);
                if (typeof temp === 'object') {
                    intermediateArr.push(temp[value]);
                } else {
                    intermediateArr.push(temp);
                }
            });
        } else {
            Object.keys(intermediateData).map(item => {
                const temp = JSON.parse(intermediateData[item].data);
                if (typeof temp === 'object') {
                    intermediateArr.push(temp[value]);
                }
            });
        }
        const intermediate = intermediateGraphOption(intermediateArr, intermediateId);
        // re-render
        this.setState({
            intermediateOption: intermediate
        });
    }

    hideIntermediateModal = () => {
        this.setState({
            modalVisible: false
        });
    }

    hideShowColumnModal = () => {
        this.setState({
            isShowColumn: false
        });
    }

    // click add column btn, just show the modal of addcolumn
    addColumn = () => {
        // show user select check button
        this.setState({
            isShowColumn: true
        });
    }

    // checkbox for coloumn
    selectedColumn = (checkedValues: Array<string>) => {
        // 9: because have nine common column, 
        // [Intermediate count, Start Time, End Time] is hidden by default
        let count = 9;
        const want: Array<object> = [];
        const finalKeys: Array<string> = [];
        const wantResult: Array<string> = [];
        Object.keys(checkedValues).map(m => {
            switch (checkedValues[m]) {
                case 'Trial No.':
                case 'ID':
                case 'Start Time':
                case 'End Time':
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

        this.props.changeColumn(wantResult);
    }

    openRow = (record: TableRecord) => {
        return (
            <OpenRow trialId={record.id} />
        );
    }

    fillSelectedRowsTostate = (selected: number[] | string[], selectedRows: Array<TableRecord>) => {
        this.setState({ selectRows: selectedRows, selectedRowKeys: selected });
    }
    // open Compare-modal
    compareBtn = () => {

        const { selectRows } = this.state;
        if (selectRows.length === 0) {
            alert('Please select datas you want to compare!');
        } else {
            this.setState({ isShowCompareModal: true });
        }
    }
    // close Compare-modal
    hideCompareModal = () => {
        // close modal. clear select rows data, clear selected track
        this.setState({ isShowCompareModal: false, selectedRowKeys: [], selectRows: [] });
    }

    // open customized trial modal
    setCustomizedTrial = (trialId: string) => {
        this.setState({
            isShowCustomizedModal: true,
            copyTrialId: trialId
        });
    }

    closeCustomizedTrial = () => {
        this.setState({
            isShowCustomizedModal: false,
            copyTrialId: ''
        });
    }
    render() {
        const { pageSize, columnList } = this.props;
        const tableSource: Array<TableRecord> = JSON.parse(JSON.stringify(this.props.tableSource));
        const { intermediateOption, modalVisible, isShowColumn,
            selectRows, isShowCompareModal, selectedRowKeys, intermediateOtherKeys,
            isShowCustomizedModal, copyTrialId
        } = this.state;
        const rowSelection = {
            selectedRowKeys: selectedRowKeys,
            onChange: (selected: string[] | number[], selectedRows: Array<TableRecord>) => {
                this.fillSelectedRowsTostate(selected, selectedRows);
            }
        };
        // [supportCustomizedTrial: true]
        const supportCustomizedTrial = (EXPERIMENT.multiPhase === true) ? false : true;
        const disabledAddCustomizedTrial = ['DONE', 'ERROR', 'STOPPED'].includes(EXPERIMENT.status);
        let showTitle = COLUMNPro;
        const showColumn: Array<object> = [];

        // parameter as table column
        const parameterStr: Array<string> = [];
        if (tableSource.length > 0) {
            const trialMess = TRIALS.getTrial(tableSource[0].id);
            const trial = trialMess.description.parameters;
            const parameterColumn: Array<string> = Object.keys(trial);
            parameterColumn.forEach(value => {
                parameterStr.push(`${value} (search space)`);
            });
        }
        showTitle = COLUMNPro.concat(parameterStr);

        // only succeed trials have final keys
        if (tableSource.filter(record => record.status === 'SUCCEEDED').length >= 1) {
            const temp = tableSource.filter(record => record.status === 'SUCCEEDED')[0].accuracy;
            if (temp !== undefined && typeof temp === 'object') {
                // concat default column and finalkeys
                const item = Object.keys(temp);
                // item: ['default', 'other-keys', 'maybe loss']
                if (item.length > 1) {
                    const want: Array<string> = [];
                    item.forEach(value => {
                        if (value !== 'default') {
                            want.push(value);
                        }
                    });
                    showTitle = COLUMNPro.concat(want);
                }
            }
        }
        for (const item of columnList) {
            const paraColumn = item.match(/ \(search space\)$/);
            let cc;
            if (paraColumn !== null) {
                cc = paraColumn.input;
            }
            switch (item) {
                case 'Trial No.':
                    showColumn.push(SequenceIdColumnConfig);
                    break;
                case 'ID':
                    showColumn.push(IdColumnConfig);
                    break;
                case 'Start Time':
                    showColumn.push(StartTimeColumnConfig);
                    break;
                case 'End Time':
                    showColumn.push(EndTimeColumnConfig);
                    break;
                case 'Duration':
                    showColumn.push(DurationColumnConfig);
                    break;
                case 'Status':
                    showColumn.push(StatusColumnConfig);
                    break;
                case 'Intermediate result':
                    showColumn.push(IntermediateCountColumnConfig);
                    break;
                case 'Default':
                    showColumn.push(AccuracyColumnConfig);
                    break;
                case 'Operation':
                    showColumn.push({
                        title: 'Operation',
                        dataIndex: 'operation',
                        key: 'operation',
                        render: (text: string, record: TableRecord) => {
                            let trialStatus = record.status;
                            const flag: boolean = (trialStatus === 'RUNNING') ? false : true;
                            return (
                                <Row id="detail-button">
                                    {/* see intermediate result graph */}
                                    <Button
                                        type="primary"
                                        className="common-style"
                                        onClick={this.showIntermediateModal.bind(this, record.id)}
                                        title="Intermediate"
                                    >
                                        <Icon type="line-chart" />
                                    </Button>
                                    {/* kill job */}
                                    {
                                        flag
                                            ?
                                            <Button
                                                type="default"
                                                disabled={true}
                                                className="margin-mediate special"
                                                title="kill"
                                            >
                                                <Icon type="stop" />
                                            </Button>
                                            :
                                            <Popconfirm
                                                title="Are you sure to cancel this trial?"
                                                okText="Yes"
                                                cancelText="No"
                                                onConfirm={killJob.
                                                    bind(this, record.key, record.id, record.status)}
                                            >
                                                <Button
                                                    type="default"
                                                    disabled={false}
                                                    className="margin-mediate special"
                                                    title="kill"
                                                >
                                                    <Icon type="stop" />
                                                </Button>
                                            </Popconfirm>
                                    }
                                    {/* Add a new trial-customized trial */}
                                    {
                                        supportCustomizedTrial
                                            ?
                                            <Button
                                                type="primary"
                                                className="common-style"
                                                disabled={disabledAddCustomizedTrial}
                                                onClick={this.setCustomizedTrial.bind(this, record.id)}
                                                title="Customized trial"
                                            >
                                                <Icon type="copy" />
                                            </Button>
                                            :
                                            null
                                    }
                                </Row>
                            );
                        },
                    });
                    break;
                case (cc):
                    // remove SEARCH_SPACE title
                    const realItem = item.replace(' (search space)', '');
                    showColumn.push({
                        title: realItem,
                        dataIndex: item,
                        key: item,
                        render: (text: string, record: TableRecord) => {
                            const eachTrial = TRIALS.getTrial(record.id);
                            return (
                                <span>{eachTrial.description.parameters[realItem]}</span>
                            );
                        },
                    });
                    break;
                default:
                    // FIXME
                    alert('Unexpected column type');
            }
        }

        return (
            <Row className="tableList">
                <div id="tableList">
                    <Table
                        ref={(table: Table<TableRecord> | null) => this.tables = table}
                        columns={showColumn}
                        rowSelection={rowSelection}
                        expandedRowRender={this.openRow}
                        dataSource={tableSource}
                        className="commonTableStyle"
                        scroll={{ x: 'max-content' }}
                        pagination={pageSize > 0 ? { pageSize } : false}
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
                        {
                            intermediateOtherKeys.length > 1
                                ?
                                <Row className="selectKeys">
                                    <Select
                                        className="select"
                                        defaultValue="default"
                                        onSelect={this.selectOtherKeys}
                                    >
                                        {
                                            Object.keys(intermediateOtherKeys).map(item => {
                                                const keys = intermediateOtherKeys[item];
                                                return <Option value={keys} key={item}>{keys}</Option>;
                                            })
                                        }
                                    </Select>

                                </Row>
                                :
                                <div />
                        }
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
                        defaultValue={columnList}
                        // defaultValue={columnSelected}
                        onChange={this.selectedColumn}
                        className="titleColumn"
                    />
                </Modal>
                {/* compare trials based message */}
                <Compare compareRows={selectRows} visible={isShowCompareModal} cancelFunc={this.hideCompareModal} />
                {/* clone trial parameters and could submit a customized trial */}
                <Customize
                    visible={isShowCustomizedModal}
                    copyTrialId={copyTrialId}
                    closeCustomizeModal={this.closeCustomizedTrial}
                />
            </Row>
        );
    }
}

const SequenceIdColumnConfig: ColumnProps<TableRecord> = {
    title: 'Trial No.',
    dataIndex: 'sequenceId',
    className: 'tableHead',
    sorter: (a, b) => a.sequenceId - b.sequenceId
};

const IdColumnConfig: ColumnProps<TableRecord> = {
    title: 'ID',
    dataIndex: 'id',
    className: 'tableHead leftTitle',
    sorter: (a, b) => a.id.localeCompare(b.id),
    render: (text, record) => (
        <div>{record.id}</div>
    )
};

const StartTimeColumnConfig: ColumnProps<TableRecord> = {
    title: 'Start Time',
    dataIndex: 'startTime',
    sorter: (a, b) => a.startTime - b.startTime,
    render: (text, record) => (
        <span>{formatTimestamp(record.startTime)}</span>
    )
};

const EndTimeColumnConfig: ColumnProps<TableRecord> = {
    title: 'End Time',
    dataIndex: 'endTime',
    sorter: (a, b, sortOrder) => {
        if (a.endTime === undefined) {
            return sortOrder === 'ascend' ? 1 : -1;
        } else if (b.endTime === undefined) {
            return sortOrder === 'ascend' ? -1 : 1;
        } else {
            return a.endTime - b.endTime;
        }
    },
    render: (text, record) => (
        <span>{formatTimestamp(record.endTime, '--')}</span>
    )
};

const DurationColumnConfig: ColumnProps<TableRecord> = {
    title: 'Duration',
    dataIndex: 'duration',
    sorter: (a, b) => a.duration - b.duration,
    render: (text, record) => (
        <span className="durationsty">{convertDuration(record.duration)}</span>
    )
};

const StatusColumnConfig: ColumnProps<TableRecord> = {
    title: 'Status',
    dataIndex: 'status',
    className: 'tableStatus',
    render: (text, record) => (
        <span className={`${record.status} commonStyle`}>{record.status}</span>
    ),
    sorter: (a, b) => a.status.localeCompare(b.status),
    filters: trialJobStatus.map(status => ({ text: status, value: status })),
    onFilter: (value, record) => (record.status === value)
};

const IntermediateCountColumnConfig: ColumnProps<TableRecord> = {
    title: 'Intermediate result',
    dataIndex: 'intermediateCount',
    sorter: (a, b) => a.intermediateCount - b.intermediateCount,
    render: (text, record) => (
        <span>{`#${record.intermediateCount}`}</span>
    )
};

const AccuracyColumnConfig: ColumnProps<TableRecord> = {
    title: 'Default metric',
    className: 'leftTitle',
    dataIndex: 'accuracy',
    sorter: (a, b, sortOrder) => {
        if (a.latestAccuracy === undefined) {
            return sortOrder === 'ascend' ? 1 : -1;
        } else if (b.latestAccuracy === undefined) {
            return sortOrder === 'ascend' ? -1 : 1;
        } else {
            return a.latestAccuracy - b.latestAccuracy;
        }
    },
    render: (text, record) => (
        <div>{record.formattedLatestAccuracy}</div>
    )
};

export default TableList;
