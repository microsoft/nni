import * as React from 'react';
import axios from 'axios';
import ReactEcharts from 'echarts-for-react';
import {
    Stack, Dropdown, DetailsList, IDetailsListProps, DetailsListLayoutMode,
    PrimaryButton, Modal, IDropdownOption, IColumn, Selection, SelectionMode, IconButton, TooltipHost
} from 'office-ui-fabric-react';
import { LineChart, blocked, copy } from '../Buttons/Icon';
import { MANAGER_IP, COLUMNPro } from '../../static/const';
import { convertDuration, formatTimestamp, intermediateGraphOption, parseMetrics } from '../../static/function';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { TableRecord, TrialJobInfo } from '../../static/interface';
import Details from '../overview/Details';
import ChangeColumnComponent from '../Modals/ChangeColumnComponent';
import Compare from '../Modals/Compare';
import KillJob from '../Modals/Killjob';
import Customize from '../Modals/CustomizedTrial';
import { contentStyles, iconButtonStyles } from '../Buttons/ModalTheme';
import '../../static/style/search.scss';
import '../../static/style/tableStatus.css';
import '../../static/style/logPath.scss';
import '../../static/style/search.scss';
import '../../static/style/table.scss';
import '../../static/style/button.scss';
import '../../static/style/openRow.scss';
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
    columnList: string[]; // user select columnKeys
    changeColumn: (val: string[]) => void;
    trialsUpdateBroadcast: number;
}

interface SortInfo {
    field: string;
    isDescend?: boolean;
}

interface TableListState {
    intermediateOption: object;
    modalVisible: boolean;
    isObjFinal: boolean;
    isShowColumn: boolean;
    selectRows: Array<any>;
    isShowCompareModal: boolean;
    selectedRowKeys: string[] | number[];
    intermediateData: Array<object>; // a trial's intermediate results (include dict)
    intermediateId: string;
    intermediateOtherKeys: string[];
    isShowCustomizedModal: boolean;
    copyTrialId: string; // user copy trial to submit a new customized trial
    isCalloutVisible: boolean; // kill job button callout [kill or not kill job window]
    intermediateKey: string; // intermeidate modal: which key is choosed.
    isExpand: boolean;
    modalIntermediateWidth: number;
    modalIntermediateHeight: number;
    tableColumns: IColumn[];
    allColumnList: string[];
    tableSourceForSort: Array<TableRecord>;
    sortMessage: SortInfo;
}

class TableList extends React.Component<TableListProps, TableListState> {

    public intervalTrialLog = 10;
    public trialId!: string;

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
            isCalloutVisible: false,
            copyTrialId: '',
            intermediateKey: 'default',
            isExpand: false,
            modalIntermediateWidth: window.innerWidth,
            modalIntermediateHeight: window.innerHeight,
            tableColumns: this.initTableColumnList(this.props.columnList),
            allColumnList: this.getAllColumnKeys(),
            tableSourceForSort: this.props.tableSource,
            sortMessage: { field: '', isDescend: false }
        };
    }

    // sort for table column
    onColumnClick = (ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
        const { tableColumns } = this.state;
        const { tableSource } = this.props;
        const newColumns: IColumn[] = tableColumns.slice();
        const currColumn: IColumn = newColumns.filter(item => getColumn.key === item.key)[0];
        newColumns.forEach((newCol: IColumn) => {
            if (newCol === currColumn) {
                currColumn.isSortedDescending = !currColumn.isSortedDescending;
                currColumn.isSorted = true;
            } else {
                newCol.isSorted = false;
                newCol.isSortedDescending = true;
            }
        });
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const newItems = this.copyAndSort(tableSource, currColumn.fieldName!, currColumn.isSortedDescending);
        this.setState({
            tableColumns: newColumns,
            tableSourceForSort: newItems,
            sortMessage: { field: getColumn.key, isDescend: currColumn.isSortedDescending }
        });

    };

    private copyAndSort<T>(items: T[], columnKey: string, isSortedDescending?: boolean): any {
        const key = columnKey as keyof T;
        return items.slice(0).sort(function (a: T, b: T): any {
            if (a[key] === undefined) {
                return 1;
            }
            if (b[key] === undefined) {
                return -1;
            }
            return (isSortedDescending ? a[key] < b[key] : a[key] > b[key]) ? 1 : -1;
        });
    }

    AccuracyColumnConfig: any = {
        name: 'Default metric',
        className: 'leftTitle',
        key: 'latestAccuracy',
        fieldName: 'latestAccuracy',
        minWidth: 200,
        maxWidth: 300,
        isResizable: true,
        data: 'number',
        onColumnClick: this.onColumnClick,
        onRender: (item): React.ReactNode => <TooltipHost content={item.formattedLatestAccuracy}>
            <div className="ellipsis">{item.formattedLatestAccuracy}</div>
        </TooltipHost>
    };

    SequenceIdColumnConfig: any = {
        name: 'Trial No.',
        key: 'sequenceId',
        fieldName: 'sequenceId',
        minWidth: 80,
        maxWidth: 240,
        className: 'tableHead',
        data: 'number',
        onColumnClick: this.onColumnClick,
    };

    IdColumnConfig: any = {
        name: 'ID',
        key: 'id',
        fieldName: 'id',
        minWidth: 150,
        maxWidth: 200,
        isResizable: true,
        data: 'string',
        onColumnClick: this.onColumnClick,
        className: 'tableHead leftTitle'
    };


    StartTimeColumnConfig: any = {
        name: 'Start Time',
        key: 'startTime',
        fieldName: 'startTime',
        minWidth: 150,
        maxWidth: 400,
        isResizable: true,
        data: 'number',
        onColumnClick: this.onColumnClick,
        onRender: (record): React.ReactNode => (
            <span>{formatTimestamp(record.startTime)}</span>
        )
    };

    EndTimeColumnConfig: any = {
        name: 'End Time',
        key: 'endTime',
        fieldName: 'endTime',
        minWidth: 200,
        maxWidth: 400,
        isResizable: true,
        data: 'number',
        onColumnClick: this.onColumnClick,
        onRender: (record): React.ReactNode => (
            <span>{formatTimestamp(record.endTime, '--')}</span>
        )
    };

    DurationColumnConfig: any = {
        name: 'Duration',
        key: 'duration',
        fieldName: 'duration',
        minWidth: 150,
        maxWidth: 300,
        isResizable: true,
        data: 'number',
        onColumnClick: this.onColumnClick,
        onRender: (record): React.ReactNode => (
            <span className="durationsty">{convertDuration(record.duration)}</span>
        )
    };

    StatusColumnConfig: any = {
        name: 'Status',
        key: 'status',
        fieldName: 'status',
        className: 'tableStatus',
        minWidth: 150,
        maxWidth: 250,
        isResizable: true,
        data: 'string',
        onColumnClick: this.onColumnClick,
        onRender: (record): React.ReactNode => (
            <span className={`${record.status} commonStyle`}>{record.status}</span>
        ),
    };

    IntermediateCountColumnConfig: any = {
        name: 'Intermediate result',
        dataIndex: 'intermediateCount',
        fieldName: 'intermediateCount',
        minWidth: 150,
        maxWidth: 200,
        isResizable: true,
        data: 'number',
        onColumnClick: this.onColumnClick,
        onRender: (record): React.ReactNode => (
            <span>{`#${record.intermediateCount}`}</span>
        )
    };

    showIntermediateModal = async (record: TrialJobInfo, event: React.SyntheticEvent<EventTarget>): Promise<void> => {
        event.preventDefault();
        event.stopPropagation();
        const res = await axios.get(`${MANAGER_IP}/metric-data/${record.jobId}`);
        if (res.status === 200) {
            const intermediateArr: number[] = [];
            // support intermediate result is dict because the last intermediate result is
            // final result in a succeed trial, it may be a dict.
            // get intermediate result dict keys array
            const { intermediateKey } = this.state;
            const otherkeys: string[] = [];
            // One trial job may contains multiple parameter id
            // only show current trial's metric data
            const metricDatas = res.data.filter(item => {
                return item.parameterId == record.parameterId;
            });
            if (metricDatas.length !== 0) {
                // just add type=number keys
                const intermediateMetrics = parseMetrics(res.data[0].data);
                for (const key in intermediateMetrics) {
                    if (typeof intermediateMetrics[key] === 'number') {
                        otherkeys.push(key);
                    }
                }
            }
            // intermediateArr just store default val
            metricDatas.map(item => {

                if (item.type === 'PERIODICAL') {
                    const temp = parseMetrics(item.data);
                    if (typeof temp === 'object') {
                        intermediateArr.push(temp[intermediateKey]);
                    } else {
                        intermediateArr.push(temp);
                    }
                }
            });
            const intermediate = intermediateGraphOption(intermediateArr, record.id);
            this.setState({
                intermediateData: res.data, // store origin intermediate data for a trial
                intermediateOption: intermediate,
                intermediateOtherKeys: otherkeys,
                intermediateId: record.id
            });
        }
        this.setState({ modalVisible: true });
    }

    // intermediate button click -> intermediate graph for each trial
    // support intermediate is dict
    selectOtherKeys = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            const value = item.text;
            const isShowDefault: boolean = value === 'default' ? true : false;
            const { intermediateData, intermediateId } = this.state;
            const intermediateArr: number[] = [];
            // just watch default key-val
            if (isShowDefault === true) {
                Object.keys(intermediateData).map(item => {
                    if (intermediateData[item].type === 'PERIODICAL') {
                        const temp = parseMetrics(intermediateData[item].data);
                        if (typeof temp === 'object') {
                            intermediateArr.push(temp[value]);
                        } else {
                            intermediateArr.push(temp);
                        }
                    }
                });
            } else {
                Object.keys(intermediateData).map(item => {
                    const temp = parseMetrics(intermediateData[item].data);
                    if (typeof temp === 'object') {
                        intermediateArr.push(temp[value]);
                    }
                });
            }
            const intermediate = intermediateGraphOption(intermediateArr, intermediateId);
            // re-render
            this.setState({
                intermediateKey: value,
                intermediateOption: intermediate
            });
        }
    }

    hideIntermediateModal = (): void => {
        this.setState({
            modalVisible: false
        });
    }

    hideShowColumnModal = (): void => {

        this.setState(() => ({ isShowColumn: false }));
    }

    // click add column btn, just show the modal of addcolumn
    addColumn = (): void => {
        // show user select check button
        this.setState(() => ({ isShowColumn: true }));
    }

    fillSelectedRowsTostate = (selected: number[] | string[], selectedRows: Array<TableRecord>): void => {
        this.setState({ selectRows: selectedRows, selectedRowKeys: selected });
    }

    // open Compare-modal
    compareBtn = (): void => {

        const { selectRows } = this.state;
        if (selectRows.length === 0) {
            alert('Please select datas you want to compare!');
        } else {
            this.setState({ isShowCompareModal: true });
        }
    }

    // close Compare-modal
    hideCompareModal = (): void => {
        // close modal. clear select rows data, clear selected track
        this.setState({ isShowCompareModal: false, selectedRowKeys: [], selectRows: [] });
    }

    // open customized trial modal
    private setCustomizedTrial = (trialId: string, event: React.SyntheticEvent<EventTarget>): void => {
        event.preventDefault();
        event.stopPropagation();
        this.setState({
            isShowCustomizedModal: true,
            copyTrialId: trialId
        });
    }

    private closeCustomizedTrial = (): void => {
        this.setState({
            isShowCustomizedModal: false,
            copyTrialId: ''
        });
    }

    private onWindowResize = (): void => {
        this.setState(() => ({
            modalIntermediateHeight: window.innerHeight,
            modalIntermediateWidth: window.innerWidth
        }));
    }

    private onRenderRow: IDetailsListProps['onRenderRow'] = props => {
        if (props) {
            return <Details detailsProps={props} />;
        }
        return null;
    };

    private getSelectedRows = new Selection({
        onSelectionChanged: (): void => {
            this.setState(() => ({ selectRows: this.getSelectedRows.getSelection() }));
        }
    });

    // trial parameters & dict final keys & Trial No. Id ...
    private getAllColumnKeys = (): string[] => {
        const tableSource: Array<TableRecord> = JSON.parse(JSON.stringify(this.props.tableSource));
        // parameter as table column
        const parameterStr: string[] = [];
        if (!EXPERIMENT.isNestedExp()) {
            if (tableSource.length > 0) {
                const trialMess = TRIALS.getTrial(tableSource[0].id);
                const trial = trialMess.description.parameters;
                const parameterColumn: string[] = Object.keys(trial);
                parameterColumn.forEach(value => {
                    parameterStr.push(`${value} (search space)`);
                });
            }
        }
        // concat trial all final keys and remove dup "default" val, return list
        const finalKeysList = TRIALS.finalKeys().filter(item => item !== 'default');
        return (COLUMNPro.concat(parameterStr)).concat(finalKeysList);
    }

    // get IColumn[]
    // when user click [Add Column] need to use the function
    private initTableColumnList = (columnList: string[]): IColumn[] => {
        // const { columnList } = this.props;
        const disabledAddCustomizedTrial = ['DONE', 'ERROR', 'STOPPED'].includes(EXPERIMENT.status);
        const showColumn: IColumn[] = [];
        for (const item of columnList) {
            const paraColumn = item.match(/ \(search space\)$/);
            let result;
            if (paraColumn !== null) {
                result = paraColumn.input;
            }
            switch (item) {
                case 'Trial No.':
                    showColumn.push(this.SequenceIdColumnConfig);
                    break;
                case 'ID':
                    showColumn.push(this.IdColumnConfig);
                    break;
                case 'Start Time':
                    showColumn.push(this.StartTimeColumnConfig);
                    break;
                case 'End Time':
                    showColumn.push(this.EndTimeColumnConfig);
                    break;
                case 'Duration':
                    showColumn.push(this.DurationColumnConfig);
                    break;
                case 'Status':
                    showColumn.push(this.StatusColumnConfig);
                    break;
                case 'Intermediate result':
                    showColumn.push(this.IntermediateCountColumnConfig);
                    break;
                case 'Default':
                    showColumn.push(this.AccuracyColumnConfig);
                    break;
                case 'Operation':
                    showColumn.push({
                        name: 'Operation',
                        key: 'operation',
                        fieldName: 'operation',
                        minWidth: 160,
                        maxWidth: 200,
                        isResizable: true,
                        className: 'detail-table',
                        onRender: (record: any) => {
                            const trialStatus = record.status;
                            const flag: boolean = (trialStatus === 'RUNNING' || trialStatus === 'UNKNOWN') ? false : true;
                            return (
                                <Stack className="detail-button" horizontal>
                                    {/* see intermediate result graph */}
                                    <PrimaryButton
                                        className="detail-button-operation"
                                        title="Intermediate"
                                        onClick={this.showIntermediateModal.bind(this, record)}
                                    >
                                        {LineChart}
                                    </PrimaryButton>
                                    {/* kill job */}
                                    {
                                        flag
                                            ?
                                            <PrimaryButton className="detail-button-operation" disabled={true} title="kill">
                                                {blocked}
                                            </PrimaryButton>
                                            :
                                            <KillJob trial={record} />
                                    }
                                    {/* Add a new trial-customized trial */}
                                    <PrimaryButton
                                        className="detail-button-operation"
                                        title="Customized trial"
                                        onClick={this.setCustomizedTrial.bind(this, record.id)}
                                        disabled={disabledAddCustomizedTrial}
                                    >
                                        {copy}
                                    </PrimaryButton>
                                </Stack>
                            );
                        },
                    });
                    break;
                case (result):
                    // remove SEARCH_SPACE title
                    // const realItem = item.replace(' (search space)', '');
                    showColumn.push({
                        name: item.replace(' (search space)', ''),
                        key: item,
                        fieldName: item,
                        minWidth: 150,
                        onRender: (record: TableRecord) => {
                            const eachTrial = TRIALS.getTrial(record.id);
                            return (
                                <span>{eachTrial.description.parameters[item.replace(' (search space)', '')]}</span>
                            );
                        },
                    });
                    break;
                default:
                    showColumn.push({
                        name: item,
                        key: item,
                        fieldName: item,
                        minWidth: 100,
                        onRender: (record: TableRecord) => {
                            const accDictionary = record.accDictionary;
                            let other = '';
                            if (accDictionary !== undefined) {
                                other = accDictionary[item].toString();
                            }
                            return (
                                <TooltipHost content={other}>
                                    <div className="ellipsis">{other}</div>
                                </TooltipHost>
                            );
                        }
                    });
            }
        }
        return showColumn;
    }

    componentDidMount(): void {
        window.addEventListener('resize', this.onWindowResize);
    }

    componentDidUpdate(prevProps: TableListProps): void {
        if (this.props.columnList !== prevProps.columnList || this.props.tableSource !== prevProps.tableSource) {
            const { columnList, tableSource } = this.props;
            this.setState({
                tableSourceForSort: tableSource,
                tableColumns: this.initTableColumnList(columnList),
                allColumnList: this.getAllColumnKeys()
            });
        }
    }

    render(): React.ReactNode {
        const { intermediateKey, modalIntermediateWidth, modalIntermediateHeight,
            tableColumns, allColumnList, isShowColumn, modalVisible,
            selectRows, isShowCompareModal, intermediateOtherKeys,
            isShowCustomizedModal, copyTrialId, intermediateOption, sortMessage
        } = this.state;
        const { columnList } = this.props;
        const tableSource: Array<TableRecord> = JSON.parse(JSON.stringify(this.state.tableSourceForSort));
        if (sortMessage.field !== '') {
            tableSource.sort(function (a, b): any {
                if (a[sortMessage.field] === undefined) {
                    return 1;
                }
                if (b[sortMessage.field] === undefined) {
                    return -1;
                }
                return (sortMessage.isDescend ? a[sortMessage.field] < b[sortMessage.field] : a[sortMessage.field] > b[sortMessage.field]) ? 1 : -1;
            });
        }
        return (
            <Stack>
                <div id="tableList">
                    <DetailsList
                        columns={tableColumns}
                        items={tableSource}
                        setKey="set"
                        compact={true}
                        onRenderRow={this.onRenderRow}
                        layoutMode={DetailsListLayoutMode.justified}
                        selectionMode={SelectionMode.multiple}
                        selection={this.getSelectedRows}
                    />

                </div>
                {/* Intermediate Result Modal */}
                <Modal
                    isOpen={modalVisible}
                    onDismiss={this.hideIntermediateModal}
                    containerClassName={contentStyles.container}
                >
                    <div className={contentStyles.header}>
                        <span>Intermediate result</span>
                        <IconButton
                            styles={iconButtonStyles}
                            iconProps={{ iconName: 'Cancel' }}
                            ariaLabel="Close popup modal"
                            onClick={this.hideIntermediateModal as any}
                        />
                    </div>
                    {
                        intermediateOtherKeys.length > 1
                            ?
                            <Stack horizontalAlign="end" className="selectKeys">
                                <Dropdown
                                    className="select"
                                    selectedKey={intermediateKey}
                                    options={
                                        intermediateOtherKeys.map((key, item) => {
                                            return {
                                                key: key, text: intermediateOtherKeys[item]
                                            };
                                        })
                                    }
                                    onChange={this.selectOtherKeys}
                                />
                            </Stack>
                            :
                            null
                    }
                    <div className="intermediate-graph">
                        <ReactEcharts
                            option={intermediateOption}
                            style={{
                                width: 0.5 * modalIntermediateWidth,
                                height: 0.7 * modalIntermediateHeight,
                                padding: 20
                            }}
                            theme="my_theme"
                        />
                        <div className="xAxis">#Intermediate result</div>
                    </div>
                </Modal>
                {/* Add Column Modal */}
                {
                    isShowColumn &&
                    <ChangeColumnComponent
                        hideShowColumnDialog={this.hideShowColumnModal}
                        isHideDialog={!isShowColumn}
                        showColumn={allColumnList}
                        selectedColumn={columnList}
                        changeColumn={this.props.changeColumn}
                    />
                }
                {/* compare trials based message */}
                {isShowCompareModal && <Compare compareStacks={selectRows} cancelFunc={this.hideCompareModal} />}
                {/* clone trial parameters and could submit a customized trial */}
                <Customize
                    visible={isShowCustomizedModal}
                    copyTrialId={copyTrialId}
                    closeCustomizeModal={this.closeCustomizedTrial}
                />
            </Stack>
        );
    }
}

export default TableList;
