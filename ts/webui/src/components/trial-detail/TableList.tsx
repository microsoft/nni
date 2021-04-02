import React from 'react';
import {
    DefaultButton,
    Dropdown,
    IColumn,
    Icon,
    IDropdownOption,
    PrimaryButton,
    Selection,
    SelectionMode,
    Stack,
    StackItem,
    TooltipHost,
    DirectionalHint,
    IContextualMenuProps
} from '@fluentui/react';
import axios from 'axios';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { TOOLTIP_BACKGROUND_COLOR, MANAGER_IP } from '../../static/const';
import { convertDuration, formatTimestamp, copyAndSort } from '../../static/function';
import { TableObj, SortInfo } from '../../static/interface';
import { blocked, copy, LineChart, tableListIcon } from '../buttons/Icon';
import ChangeColumnComponent from '../modals/ChangeColumnComponent';
import Compare from '../modals/Compare';
import Customize from '../modals/CustomizedTrial';
import Tensorboard from '../modals/tensorboard/Tensorboard';
import ShowTensorBoardDetail from '../modals/tensorboard/ShowTensorBoardDetail';
import KillJob from '../modals/Killjob';
import ExpandableDetails from '../public-child/ExpandableDetails';
import PaginationTable from '../public-child/PaginationTable';
import CopyButton from '../public-child/CopyButton';
import { Trial } from '../../static/model/trial';
// import DialogDetail from '../modals/tensorboard/RepeatTensorDialog';
import '../../static/style/button.scss';
import '../../static/style/logPath.scss';
import '../../static/style/openRow.scss';
import '../../static/style/pagination.scss';
import '../../static/style/search.scss';
import '../../static/style/table.scss';
import '../../static/style/tableStatus.css';
import '../../static/style/tensorboard.scss';
import '../../static/style/overview/overviewTitle.scss';

require('echarts/lib/chart/line');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface Tensorboard {
    id: string;
    status: string;
    trialJobIdList: string[];
    trialLogDirectoryList: string[];
    pid: number;
    port: string;
};

type SearchOptionType = 'id' | 'trialnum' | 'status' | 'parameters';
const searchOptionLiterals = {
    id: 'ID',
    trialnum: 'Trial No.',
    status: 'Status',
    parameters: 'Parameters'
};

const defaultDisplayedColumns = ['sequenceId', 'id', 'duration', 'status', 'latestAccuracy'];

function _inferColumnTitle(columnKey: string): string {
    if (columnKey === 'sequenceId') {
        return 'Trial No.';
    } else if (columnKey === 'id') {
        return 'ID';
    } else if (columnKey === 'intermediateCount') {
        return 'Intermediate results (#)';
    } else if (columnKey === 'message') {
        return 'Message';
    } else if (columnKey.startsWith('space/')) {
        return columnKey.split('/', 2)[1] + ' (space)';
    } else if (columnKey === 'latestAccuracy') {
        return 'Default metric'; // to align with the original design
    } else if (columnKey.startsWith('metric/')) {
        return columnKey.split('/', 2)[1] + ' (metric)';
    } else if (columnKey.startsWith('_')) {
        return columnKey;
    } else {
        // camel case to verbose form
        const withSpace = columnKey.replace(/[A-Z]/g, letter => ` ${letter.toLowerCase()}`);
        return withSpace.charAt(0).toUpperCase() + withSpace.slice(1);
    }
}

interface TableListProps {
    tableSource: TableObj[];
    trialsUpdateBroadcast: number;
}

interface TableListState {
    displayedItems: any[];
    displayedColumns: string[];
    columns: IColumn[];
    searchType: SearchOptionType;
    searchText: string;
    selectedRowIds: string[];
    customizeColumnsDialogVisible: boolean;
    compareDialogVisible: boolean;
    tensorboardPanelVisible: boolean;
    detailTensorboardPanelVisible: boolean;
    intermediateDialogTrial: TableObj | undefined;
    copiedTrialId: string | undefined;
    sortInfo: SortInfo;
    visibleDialog: boolean;
    dialogContent: string;
    isReaptedTensorboard: boolean;
    queryTensorboardList: Tensorboard[];
    selectedTensorboard?: Tensorboard;
}

class TableList extends React.Component<TableListProps, TableListState> {
    private _selection: Selection;
    private _expandedTrialIds: Set<string>;
    private refreshTensorboard!: number | undefined;
    private tableListComponent: boolean = false;

    constructor(props: TableListProps) {
        super(props);

        this.state = {
            displayedItems: [],
            displayedColumns:
                localStorage.getItem('columns') !== null
                    ? // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    JSON.parse(localStorage.getItem('columns')!)
                    : defaultDisplayedColumns,
            columns: [],
            searchType: 'id',
            searchText: '',
            customizeColumnsDialogVisible: false,
            compareDialogVisible: false,
            tensorboardPanelVisible: false,
            detailTensorboardPanelVisible: false,
            selectedRowIds: [],
            intermediateDialogTrial: undefined,
            copiedTrialId: undefined,
            sortInfo: { field: '', isDescend: true },
            visibleDialog: false,
            dialogContent: '',
            isReaptedTensorboard: false,
            queryTensorboardList: []
        };

        this._selection = new Selection({
            onSelectionChanged: (): void => {
                this.setState({
                    selectedRowIds: this._selection.getSelection().map(s => (s as any).id)
                });
            }
        });

        this._expandedTrialIds = new Set<string>();
    }

    /* Search related methods */

    // This functions as the filter for the final trials displayed in the current table
    private _filterTrials(trials: TableObj[]): TableObj[] {
        const { searchText, searchType } = this.state;
        // search a trial by Trial No. | Trial ID | Parameters | Status
        let searchFilter = (_: TableObj): boolean => true; // eslint-disable-line no-unused-vars
        if (searchText.trim()) {
            if (searchType === 'id') {
                searchFilter = (trial): boolean => trial.id.toUpperCase().includes(searchText.toUpperCase());
            } else if (searchType === 'trialnum') {
                searchFilter = (trial): boolean => trial.sequenceId.toString() === searchText;
            } else if (searchType === 'status') {
                searchFilter = (trial): boolean => trial.status.toUpperCase().includes(searchText.toUpperCase());
            } else if (searchType === 'parameters') {
                // TODO: support filters like `x: 2` (instead of `'x': 2`)
                searchFilter = (trial): boolean => JSON.stringify(trial.description.parameters).includes(searchText);
            }
        }
        return trials.filter(searchFilter);
    }

    private _updateSearchFilterType(_event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void {
        if (item !== undefined) {
            const value = item.key.toString();
            if (searchOptionLiterals.hasOwnProperty(value)) {
                this.setState({ searchType: value as SearchOptionType }, this._updateTableSource);
            }
        }
    }

    private _updateSearchText(ev: React.ChangeEvent<HTMLInputElement>): void {
        this.setState({ searchText: ev.target.value }, this._updateTableSource);
    }

    /* Table basic function related methods */

    private _onColumnClick(ev: React.MouseEvent<HTMLElement>, column: IColumn): void {
        // handle the click events on table header (do sorting)
        const { columns } = this.state;
        const newColumns: IColumn[] = columns.slice();
        const currColumn: IColumn = newColumns.filter(currCol => column.key === currCol.key)[0];
        const isSortedDescending = !currColumn.isSortedDescending;
        this.setState(
            {
                sortInfo: { field: column.key, isDescend: isSortedDescending }
            },
            this._updateTableSource
        );
    }

    private _trialsToTableItems(trials: TableObj[]): any[] {
        // TODO: use search space and metrics space from TRIALS will cause update issues.
        const searchSpace = TRIALS.inferredSearchSpace(EXPERIMENT.searchSpaceNew);
        const metricSpace = TRIALS.inferredMetricSpace();
        const items = trials.map(trial => {
            const ret = {
                sequenceId: trial.sequenceId,
                id: trial.id,
                startTime: (trial as Trial).info.startTime, // FIXME: why do we need info here?
                endTime: (trial as Trial).info.endTime,
                duration: trial.duration,
                status: trial.status,
                message: (trial as Trial).info.message || '--',
                intermediateCount: trial.intermediates.length,
                _expandDetails: this._expandedTrialIds.has(trial.id) // hidden field names should start with `_`
            };
            for (const [k, v] of trial.parameters(searchSpace)) {
                ret[`space/${k.baseName}`] = v;
            }
            for (const [k, v] of trial.metrics(metricSpace)) {
                ret[`metric/${k.baseName}`] = v;
            }
            ret['latestAccuracy'] = (trial as Trial).latestAccuracy;
            ret['_formattedLatestAccuracy'] = (trial as Trial).formatLatestAccuracy();
            return ret;
        });

        const { sortInfo } = this.state;
        if (sortInfo.field !== '') {
            return copyAndSort(items, sortInfo.field, sortInfo.isDescend);
        } else {
            return items;
        }
    }

    private _buildColumnsFromTableItems(tableItems: any[]): IColumn[] {
        // extra column, for a icon to expand the trial details panel
        const columns: IColumn[] = [
            {
                key: '_expand',
                name: '',
                onRender: (item): any => {
                    return (
                        <Icon
                            aria-hidden={true}
                            iconName='ChevronRight'
                            className='cursor'
                            styles={{
                                root: {
                                    transition: 'all 0.2s',
                                    transform: `rotate(${item._expandDetails ? 90 : 0}deg)`
                                }
                            }}
                            onClick={(event): void => {
                                event.stopPropagation();
                                const newItem: any = { ...item, _expandDetails: !item._expandDetails };
                                if (newItem._expandDetails) {
                                    // preserve to be restored when refreshed
                                    this._expandedTrialIds.add(newItem.id);
                                } else {
                                    this._expandedTrialIds.delete(newItem.id);
                                }
                                const newItems = this.state.displayedItems.map(item =>
                                    item.id === newItem.id ? newItem : item
                                );
                                this.setState({
                                    displayedItems: newItems
                                });
                            }}
                            onMouseDown={(e): void => {
                                e.stopPropagation();
                            }}
                            onMouseUp={(e): void => {
                                e.stopPropagation();
                            }}
                        />
                    );
                },
                fieldName: 'expand',
                isResizable: false,
                minWidth: 20,
                maxWidth: 20
            }
        ];
        // looking at the first row only for now
        for (const k of Object.keys(tableItems[0])) {
            if (k === 'metric/default') {
                // FIXME: default metric is hacked as latestAccuracy currently
                continue;
            }
            const columnTitle = _inferColumnTitle(k);
            // TODO: add blacklist
            // 0.85: tableWidth / screen
            const widths = window.innerWidth * 0.85;
            columns.push({
                name: columnTitle,
                key: k,
                fieldName: k,
                minWidth: widths * 0.12,
                maxWidth: widths * 0.19,
                isResizable: true,
                onColumnClick: this._onColumnClick.bind(this),
                ...(k === 'status' && {
                    // color status
                    onRender: (record): React.ReactNode => (
                        <span className={`${record.status} commonStyle`}>{record.status}</span>
                    )
                }),
                ...(k === 'message' && {
                    onRender: (record): React.ReactNode =>
                        record.message.length > 15 ? (
                            <TooltipHost
                                content={record.message}
                                directionalHint={DirectionalHint.bottomCenter}
                                tooltipProps={{
                                    calloutProps: {
                                        styles: {
                                            beak: { background: TOOLTIP_BACKGROUND_COLOR },
                                            beakCurtain: { background: TOOLTIP_BACKGROUND_COLOR },
                                            calloutMain: { background: TOOLTIP_BACKGROUND_COLOR }
                                        }
                                    }
                                }}
                            >
                                <div>{record.message}</div>
                            </TooltipHost>
                        ) : (
                            <div>{record.message}</div>
                        )
                }),
                ...((k.startsWith('metric/') || k.startsWith('space/')) && {
                    // show tooltip
                    onRender: (record): React.ReactNode => (
                        <TooltipHost
                            content={record[k]}
                            directionalHint={DirectionalHint.bottomCenter}
                            tooltipProps={{
                                calloutProps: {
                                    styles: {
                                        beak: { background: TOOLTIP_BACKGROUND_COLOR },
                                        beakCurtain: { background: TOOLTIP_BACKGROUND_COLOR },
                                        calloutMain: { background: TOOLTIP_BACKGROUND_COLOR }
                                    }
                                }
                            }}
                        >
                            <div className='ellipsis'>{record[k]}</div>
                        </TooltipHost>
                    )
                }),
                ...(k === 'latestAccuracy' && {
                    // FIXME: this is ad-hoc
                    onRender: (record): React.ReactNode => (
                        <TooltipHost
                            content={record._formattedLatestAccuracy}
                            directionalHint={DirectionalHint.bottomCenter}
                            tooltipProps={{
                                calloutProps: {
                                    styles: {
                                        beak: { background: TOOLTIP_BACKGROUND_COLOR },
                                        beakCurtain: { background: TOOLTIP_BACKGROUND_COLOR },
                                        calloutMain: { background: TOOLTIP_BACKGROUND_COLOR }
                                    }
                                }
                            }}
                        >
                            <div className='ellipsis'>{record._formattedLatestAccuracy}</div>
                        </TooltipHost>
                    )
                }),
                ...(['startTime', 'endTime'].includes(k) && {
                    onRender: (record): React.ReactNode => <span>{formatTimestamp(record[k], '--')}</span>
                }),
                ...(k === 'duration' && {
                    onRender: (record): React.ReactNode => (
                        <span className='durationsty'>{convertDuration(record[k])}</span>
                    )
                }),
                ...(k === 'id' && {
                    onRender: (record): React.ReactNode => (
                        <Stack horizontal className='idCopy'>
                            <div>{record.id}</div>
                            <CopyButton value={record.id} />
                        </Stack>
                    )
                })
            });
        }
        // operations column
        columns.push({
            name: 'Operation',
            key: '_operation',
            fieldName: 'operation',
            minWidth: 150,
            maxWidth: 160,
            isResizable: true,
            className: 'detail-table',
            onRender: this._renderOperationColumn.bind(this)
        });

        const { sortInfo } = this.state;
        for (const column of columns) {
            if (column.key === sortInfo.field) {
                column.isSorted = true;
                column.isSortedDescending = sortInfo.isDescend;
            } else {
                column.isSorted = false;
                column.isSortedDescending = true;
            }
        }
        return columns;
    }

    private _updateTableSource(): void {
        // call this method when trials or the computation of trial filter has changed
        const items = this._trialsToTableItems(this._filterTrials(this.props.tableSource));
        if (items.length > 0) {
            const columns = this._buildColumnsFromTableItems(items);
            this.setState({
                displayedItems: items,
                columns: columns
            });
        } else {
            this.setState({
                displayedItems: [],
                columns: []
            });
        }
    }

    private _updateDisplayedColumns(displayedColumns: string[]): void {
        this.setState({
            displayedColumns: displayedColumns
        });
    }

    private _renderOperationColumn(record: any): React.ReactNode {
        const runningTrial: boolean = ['RUNNING', 'UNKNOWN'].includes(record.status) ? false : true;
        const disabledAddCustomizedTrial = ['DONE', 'ERROR', 'STOPPED'].includes(EXPERIMENT.status);
        return (
            <Stack className='detail-button' horizontal>
                <PrimaryButton
                    className='detail-button-operation'
                    title='Intermediate'
                    onClick={(): void => {
                        const { tableSource } = this.props;
                        const trial = tableSource.find(trial => trial.id === record.id) as TableObj;
                        this.setState({ intermediateDialogTrial: trial });
                    }}
                >
                    {LineChart}
                </PrimaryButton>
                {runningTrial ? (
                    <PrimaryButton className='detail-button-operation' disabled={true} title='kill'>
                        {blocked}
                    </PrimaryButton>
                ) : (
                    <KillJob trial={record} />
                )}
                <PrimaryButton
                    className='detail-button-operation'
                    title='Customized trial'
                    onClick={(): void => {
                        this.setState({ copiedTrialId: record.id });
                    }}
                    disabled={disabledAddCustomizedTrial}
                >
                    {copy}
                </PrimaryButton>
            </Stack>
        );
    }

    private clearAllTensorboard = (): void => {
        // clear all 所有状态
        axios(`${MANAGER_IP}/tensorboard-tasks`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
        })
            .then(res => {
                if (res.status === 200) {
                    // 成功停掉了所有在运行的tensorboard，状态一律称为 stopped
                    // 提示所有的tensorboard都已经停掉了，清掉所有的query, 关掉定时器
                    this.setState(() => ({ queryTensorboardList: [] }));
                    window.clearTimeout(this.refreshTensorboard);
                }
            })
    }

    private queryAllTensorboard = (): void => {
        // 查询所有trial tensorboard 状态
        // TODO: 加error场景
        if(this.tableListComponent){
            axios(`${MANAGER_IP}/tensorboard-tasks`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
            })
                .then(res => {
                    if (res.status === 200) {
                        this.setState(() => ({ queryTensorboardList: res.data.filter(item => item.status !== 'STOPPED') }));
                    }
                })
                .catch(error => {
                    console.info(error);
                });
            this.refreshTensorboard = window.setTimeout(this.queryAllTensorboard, 10000);
        }
    }

    private startTrialTensorboard = (): void => {
        const { selectedRowIds } = this.state;
        // 查询所有 trial tensorboard 状态
        axios(`${MANAGER_IP}/tensorboard-tasks`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
        })
            .then(res => {
                if (res.status === 200) {
                    const data = res.data;
                    // ??? 程序一开始空的时候
                    const result = data.filter(item => item.status !== 'STOPPED' && item.trialJobIdList.join(',') === selectedRowIds.join(','));
                    if (result.length > 0) {
                        this.setState({ isReaptedTensorboard: true, selectedTensorboard: result[0], tensorboardPanelVisible: true });
                    } else {
                        axios(`${MANAGER_IP}/tensorboard`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            data: {
                                trials: selectedRowIds.join(',')
                            }
                        })
                            .then(res => {
                                if (res.status === 200) {

                                    // setTimeout((): void => {
                                    this.setState({ isReaptedTensorboard: false, selectedTensorboard: res.data, tensorboardPanelVisible: true });
                                    // }, 500);
                                    // 每10s刷一次表单数据
                                    // 如果表单没显示出来也会默默刷新
                                    this.queryAllTensorboard();
                                }
                            })
                            .catch(error => {
                                this.setState({ isReaptedTensorboard: false, visibleDialog: true, dialogContent: error.message || 'Tensorboard start failed' });
                            });
                    }
                }
            })
    }

    private closeDialog = (): void => {
        this.setState({ visibleDialog: false });
    }

    private seeTensorboardWebportal = (item: Tensorboard): void => {
        // 加弹窗再提醒一下消息
        this.setState({ detailTensorboardPanelVisible: true, selectedTensorboard:  item});
    }

    componentDidUpdate(prevProps: TableListProps): void {
        if (this.props.tableSource !== prevProps.tableSource) {
            this._updateTableSource();
        }
    }

    componentDidMount(): void {
        this.tableListComponent = true;
        this._updateTableSource();
        this.queryAllTensorboard();
    }

    componentWillUnmount(): void {
        this.tableListComponent = false;
        window.clearTimeout(this.refreshTensorboard);
    }

    render(): React.ReactNode {
        const {
            displayedItems,
            columns,
            searchType,
            customizeColumnsDialogVisible,
            compareDialogVisible,
            tensorboardPanelVisible,
            detailTensorboardPanelVisible,
            displayedColumns,
            selectedRowIds,
            intermediateDialogTrial,
            copiedTrialId,
            // visibleDialog,
            // dialogContent,
            isReaptedTensorboard,
            queryTensorboardList,
            selectedTensorboard
        } = this.state;
        const some: Array<object> = [];
        if (queryTensorboardList.length !== 0) {
            some.push({
                key: 'delete',
                text: 'Stop all tensorBoard',
                className: 'clearAll',
                onClick: this.clearAllTensorboard
            });
            queryTensorboardList.forEach(item => {
                some.push({
                    key: item.id,
                    text: `${item.id} ${item.port}`,
                    className: `CommandBarButton-${item.status}`,
                    onClick: this.seeTensorboardWebportal.bind(this, item)
                });
            })
            
        }
        const tensorboardMenu: IContextualMenuProps = {
            items: some.reverse() as any
        };
        // disable tensorboard btn logic
        let flag = true;
        if (selectedRowIds.length !== 0) {
            flag = false;
        }
            
        if (selectedRowIds.length === 0 && queryTensorboardList.length !== 0) {
            flag = false;
        }
        return (
            <div id='tableList'>
                <Stack horizontal className='panelTitle' style={{ marginTop: 10 }}>
                    <span style={{ marginRight: 12 }}>{tableListIcon}</span>
                    <span>Trial jobs</span>
                </Stack>
                <Stack horizontal className='allList'>
                    <StackItem grow={50}>
                        <DefaultButton
                            text='Compare'
                            className='allList-compare'
                            onClick={(): void => {
                                this.setState({ compareDialogVisible: true });
                            }}
                            disabled={selectedRowIds.length === 0}
                        />
                        <DefaultButton
                            text='TensorBoard'
                            className='elementMarginLeft'
                            split
                            splitButtonAriaLabel="See 2 options"
                            aria-roledescription="split button"
                            menuProps={tensorboardMenu}
                            onClick={(): void => this.startTrialTensorboard()}
                            disabled={flag}
                        />
                        {
                            queryTensorboardList.length !== 0 ?
                                <span className='circle'>{queryTensorboardList.length}</span>
                                : null
                        }

                    </StackItem>
                    <StackItem grow={50}>
                        <Stack horizontal horizontalAlign='end' className='allList'>
                            <DefaultButton
                                className='allList-button-gap'
                                text='Add/Remove columns'
                                onClick={(): void => {
                                    this.setState({ customizeColumnsDialogVisible: true });
                                }}
                            />
                            <Dropdown
                                selectedKey={searchType}
                                options={Object.entries(searchOptionLiterals).map(([k, v]) => ({
                                    key: k,
                                    text: v
                                }))}
                                onChange={this._updateSearchFilterType.bind(this)}
                                styles={{ root: { width: 150 } }}
                            />
                            <input
                                type='text'
                                className='allList-search-input'
                                placeholder={`Search by ${['id', 'trialnum'].includes(searchType)
                                    ? searchOptionLiterals[searchType]
                                    : searchType
                                    }`}
                                onChange={this._updateSearchText.bind(this)}
                                style={{ width: 230 }}
                            />
                        </Stack>
                    </StackItem>
                </Stack>
                {columns && displayedItems && (
                    <PaginationTable
                        columns={columns.filter(
                            column =>
                                displayedColumns.includes(column.key) || ['_expand', '_operation'].includes(column.key)
                        )}
                        items={displayedItems}
                        compact={true}
                        selection={this._selection}
                        selectionMode={SelectionMode.multiple}
                        selectionPreservedOnEmptyClick={true}
                        onRenderRow={(props): any => {
                            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                            return <ExpandableDetails detailsProps={props!} isExpand={props!.item._expandDetails} />;
                        }}
                    />
                )}
                {compareDialogVisible && (
                    <Compare
                        title='Compare trials'
                        showDetails={true}
                        trials={this.props.tableSource.filter(trial => selectedRowIds.includes(trial.id))}
                        onHideDialog={(): void => {
                            this.setState({ compareDialogVisible: false });
                        }}
                    />
                )}
                {tensorboardPanelVisible && (
                    <Tensorboard
                        isReaptedTensorboard={isReaptedTensorboard}
                        item={selectedTensorboard}
                        onHideDialog={(): void => {
                            this.setState({ tensorboardPanelVisible: false });
                        }}
                    />
                )}
                {detailTensorboardPanelVisible && (
                    <ShowTensorBoardDetail
                        item={selectedTensorboard}
                        onHideDialog={(): void => {
                            this.setState({ detailTensorboardPanelVisible: false });
                        }}
                    />
                )}
                {intermediateDialogTrial !== undefined && (
                    <Compare
                        title='Intermediate results'
                        showDetails={false}
                        trials={[intermediateDialogTrial]}
                        onHideDialog={(): void => {
                            this.setState({ intermediateDialogTrial: undefined });
                        }}
                    />
                )}
                {customizeColumnsDialogVisible && (
                    <ChangeColumnComponent
                        selectedColumns={displayedColumns}
                        allColumns={columns
                            .filter(column => !column.key.startsWith('_'))
                            .map(column => ({ key: column.key, name: column.name }))}
                        onSelectedChange={this._updateDisplayedColumns.bind(this)}
                        onHideDialog={(): void => {
                            this.setState({ customizeColumnsDialogVisible: false });
                        }}
                    />
                )}
                {/* Clone a trial and customize a set of new parameters */}
                {/* visible is done inside because prompt is needed even when the dialog is closed */}
                <Customize
                    visible={copiedTrialId !== undefined}
                    copyTrialId={copiedTrialId || ''}
                    closeCustomizeModal={(): void => {
                        this.setState({ copiedTrialId: undefined });
                    }}
                />
                {/* for all trials tensorboard failed modal */}
                {/* {visibleDialog &&
                    <DialogDetail
                        message={dialogContent}
                        func={this.closeDialog}
                    />} */}
            </div>
        );
    }
}

export default TableList;
