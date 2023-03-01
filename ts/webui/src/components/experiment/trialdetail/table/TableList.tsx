import React, { useState, useEffect, useContext } from 'react';
import { DefaultButton, IColumn, Icon, PrimaryButton, Stack, Checkbox } from '@fluentui/react';
import { Trial } from '@model/trial';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { convertDuration, formatTimestamp, copyAndSort, parametersType, _inferColumnTitle } from '@static/function';
import { SortInfo, SearchItems } from '@static/interface';
import { blocked, copy, LineChart, tableListIcon } from '@components/fluent/Icon';
import Customize from './tableFunction/CustomizedTrial';
import TensorboardUI from './tableFunction/tensorboard/TensorboardUI';
import Search from './tableFunction/search/Search';
import ChangeColumnComponent from '../ChangeColumnComponent';
import Compare from './tableFunction/CompareIndex';
import KillJobIndex from './tableFunction/killJob/KillJobIndex';
import { getTrialsBySearchFilters } from './tableFunction/search/searchFunction';
import ExpandableDetails from '@components/common/ExpandableDetails/ExpandableIndex';
import PaginationTable from '@components/common/PaginationTable';
import CopyButton from '@components/common/CopyButton';
import TooltipHostIndex from '@components/common/TooltipHostIndex';
import { buttonsGap } from '@components/common/Gap';
import { getValue } from '@model/localStorage';
import { AppContext } from '@/App';
require('echarts/lib/chart/line');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

const defaultDisplayedColumns = ['sequenceId', 'id', 'duration', 'status', 'latestAccuracy'];
const columnsWidths = [
    { name: 'sequenceId', value: [140, 250] },
    { name: 'id', value: [145, 270] },
    { name: 'duration', value: [163, 296] },
    { name: 'status', value: [165, 310] },
    { name: 'latestAccuracy', value: [180, 306] }
];
interface TableListProps {
    tableSource: Trial[];
}

const TableList = (props: TableListProps): any => {
    const { tableSource } = props;
    const { expandRowIDsDetailTable, changeExpandRowIDsDetailTable, selectedRowIds, changeSelectedRowIds } =
        useContext(AppContext);
    const [displayedColumns, setDisplayedColumns] = useState(
        localStorage.getItem(`${EXPERIMENT.profile.id}_columns`) !== null &&
            getValue(`${EXPERIMENT.profile.id}_columns`) !== null
            ? // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
              JSON.parse(getValue(`${EXPERIMENT.profile.id}_columns`)!)
            : defaultDisplayedColumns
    );
    const [columns, setColumns] = useState([] as IColumn[]);
    const [customizeColumnsDialogVisible, setCustomizeColumnsDialogVisible] = useState(false);
    const [compareDialogVisible, setCompareDialogVisible] = useState(false);
    const [intermediateDialogTrial, setIntermediateDialogTrial] = useState([] as Trial[]);
    const [copiedTrialId, setCopiedTrialId] = useState(undefined);
    const [sortInfo, setSortInfo] = useState({ field: '', isDescend: true } as SortInfo);
    const [searchItems, setSearchItems] = useState([] as SearchItems[]);
    const relation = parametersType();
    const [displayedItems, setDisplayedItems] = useState([] as any);

    /* Table basic function related methods */

    const _onColumnClick = (ev: React.MouseEvent<HTMLElement>, column: IColumn): void => {
        // handle the click events on table header (do sorting)
        const newColumns: IColumn[] = columns.slice();
        const currColumn: IColumn = newColumns.filter(currCol => column.key === currCol.key)[0];
        const isSortedDescending = !currColumn.isSortedDescending;
        setSortInfo({ field: column.key, isDescend: isSortedDescending });
    };

    const _trialsToTableItems = (trials: Trial[]): any[] => {
        // TODO: use search space and metrics space from TRIALS will cause update issues.
        const searchSpace = TRIALS.inferredSearchSpace(EXPERIMENT.searchSpaceNew);
        const metricSpace = TRIALS.inferredMetricSpace();
        const items = trials.map(trial => {
            const ret = trial.tableRecord;
            ret['_checked'] = selectedRowIds.includes(trial.id) ? true : false;
            ret['_expandDetails'] = expandRowIDsDetailTable.has(trial.id); // hidden field names should start with `_`
            for (const [k, v] of trial.parameters(searchSpace)) {
                ret[`space/${k.baseName}`] = v;
            }
            for (const [k, v] of trial.metrics(metricSpace)) {
                ret[`metric/${k.baseName}`] = v;
            }
            return ret;
        });

        if (sortInfo.field !== '') {
            return copyAndSort(items, sortInfo.field, sortInfo.isDescend);
        } else {
            return items;
        }
    };

    const changeSelectTrialIds = (): void => {
        changeSelectedRowIds([]);
    };

    const _renderOperationColumn = (record: any): React.ReactNode => {
        const runningTrial: boolean = ['RUNNING', 'UNKNOWN'].includes(record.status) ? false : true;
        const disabledAddCustomizedTrial = ['DONE', 'ERROR', 'STOPPED', 'VIEWED'].includes(EXPERIMENT.status);
        return (
            <Stack className='detail-button' horizontal>
                <PrimaryButton
                    className='detail-button-operation'
                    title='Intermediate'
                    onClick={(): void => {
                        const trial = tableSource.find(trial => trial.id === record.id) as Trial;
                        setIntermediateDialogTrial([trial]);
                    }}
                >
                    {LineChart}
                </PrimaryButton>
                {runningTrial ? (
                    <PrimaryButton className='detail-button-operation' disabled={true} title='kill'>
                        {blocked}
                    </PrimaryButton>
                ) : (
                    <KillJobIndex trialId={record.id} />
                )}
                <PrimaryButton
                    className='detail-button-operation'
                    title='Customized trial'
                    onClick={(): void => {
                        setCopiedTrialId(record.id);
                    }}
                    disabled={disabledAddCustomizedTrial}
                >
                    {copy}
                </PrimaryButton>
            </Stack>
        );
    };

    const _buildColumnsFromTableItems = (tableItems: any[]): IColumn[] => {
        const columns: IColumn[] = [
            // select trial function
            {
                name: '',
                key: '_selected',
                fieldName: 'selected',
                minWidth: 20,
                maxWidth: 20,
                isResizable: true,
                className: 'detail-table',
                onRender: (record): React.ReactNode => (
                    <Checkbox
                        label={undefined}
                        checked={record._checked}
                        className='detail-check'
                        onChange={(_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean): void => {
                            let latestSelectedRowIds = selectedRowIds;

                            if (checked === false) {
                                latestSelectedRowIds = latestSelectedRowIds.filter(item => item !== record.id);
                            } else {
                                latestSelectedRowIds.push(record.id);
                            }
                            changeSelectedRowIds(latestSelectedRowIds);
                        }}
                    />
                )
            },
            // extra column, for a icon to expand the trial details panel
            {
                key: '_expand',
                name: '',
                onRender: (item): any => {
                    return (
                        <Icon
                            aria-hidden={true}
                            iconName='ChevronRight'
                            className='cursor bold positionTop'
                            styles={{
                                root: {
                                    transition: 'all 0.2s',
                                    transform: `rotate(${expandRowIDsDetailTable.has(item.id) ? 90 : 0}deg)`
                                }
                            }}
                            onClick={(event): void => {
                                event.stopPropagation();
                                changeExpandRowIDsDetailTable(item.id);
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
            columns.push({
                name: columnTitle,
                key: k,
                fieldName: k,
                minWidth:
                    columnsWidths.find(item => item.name === k) !== undefined
                        ? columnsWidths.find(item => item.name === k)!.value[0]
                        : 150,
                maxWidth:
                    columnsWidths.find(item => item.name === k) !== undefined
                        ? columnsWidths.find(item => item.name === k)!.value[1]
                        : 250,
                isResizable: true,
                onColumnClick: _onColumnClick,
                ...(k === 'status' && {
                    // color status
                    onRender: (record): React.ReactNode => (
                        <span className={`${record.status} commonStyle`}>{record.status}</span>
                    )
                }),
                ...(k === 'message' && {
                    onRender: (record): React.ReactNode => <TooltipHostIndex value={record.message} />
                }),
                ...((k.startsWith('metric/') || k.startsWith('space/')) && {
                    // show tooltip
                    onRender: (record): React.ReactNode => <TooltipHostIndex value={record[k]} />
                }),
                ...(k === 'latestAccuracy' && {
                    // FIXME: this is ad-hoc
                    onRender: (record): React.ReactNode => <TooltipHostIndex value={record._formattedLatestAccuracy} />
                }),
                ...(['startTime', 'endTime'].includes(k) && {
                    onRender: (record): React.ReactNode => <span>{formatTimestamp(record[k], '--')}</span>
                }),
                ...(k === 'duration' && {
                    onRender: (record): React.ReactNode => <span>{convertDuration(record[k])}</span>
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
            minWidth: 207,
            maxWidth: 221,
            isResizable: true,
            className: 'detail-table',
            onRender: _renderOperationColumn
        });

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
    };

    const _updateTableSource = (): void => {
        // call this method when trials or the computation of trial filter has changed
        let items = _trialsToTableItems(tableSource);
        if (searchItems.length > 0) {
            items = getTrialsBySearchFilters(items, searchItems, relation); // use search filter to filter data
        }
        if (items.length > 0) {
            const columns = _buildColumnsFromTableItems(items);
            setColumns(columns);
            setDisplayedItems(items);
        } else {
            setColumns([]);
            setDisplayedItems([]);
        }
    };

    const _updateDisplayedColumns = (value: string[]): void => {
        setDisplayedColumns(value);
    };

    const changeSearchFilterList = (arr: Array<SearchItems>): void => {
        setSearchItems(arr);
    };

    useEffect(() => {
        _updateTableSource();
        // { sortInfo.field, sortInfo.isDescend }, displayedItems will cause endless loop
    }, [tableSource, selectedRowIds, searchItems, sortInfo]);

    return (
        <div id='tableList'>
            <Stack horizontal className='panelTitle' style={{ marginTop: 10 }}>
                <span style={{ marginRight: 12 }}>{tableListIcon}</span>
                <span className='fontColor333'>Trial jobs</span>
            </Stack>
            <Stack horizontal horizontalAlign='space-between' className='allList'>
                <Search
                    searchFilter={searchItems} // search filter list
                    changeSearchFilterList={changeSearchFilterList}
                />
                <Stack horizontal horizontalAlign='end' tokens={buttonsGap}>
                    <DefaultButton
                        text='Add/Remove columns'
                        onClick={(): void => {
                            setCustomizeColumnsDialogVisible(true);
                        }}
                    />
                    <DefaultButton
                        text='Compare'
                        onClick={(): void => {
                            setCompareDialogVisible(true);
                        }}
                        disabled={selectedRowIds.length === 0}
                    />
                    {/* compare model: trial intermediates graph; table: id,no,status,default dict value */}
                    {compareDialogVisible && (
                        <Compare
                            title='Compare trials'
                            trials={tableSource.filter(trial => selectedRowIds.includes(trial.id))}
                            onHideDialog={(): void => {
                                setCompareDialogVisible(false);
                            }}
                            changeSelectTrialIds={changeSelectTrialIds}
                        />
                    )}
                    <TensorboardUI selectedRowIds={selectedRowIds} changeSelectTrialIds={changeSelectTrialIds} />
                </Stack>
            </Stack>
            {columns && displayedItems && (
                <PaginationTable
                    columns={columns.filter(
                        column =>
                            displayedColumns.includes(column.key) ||
                            ['_expand', '_operation', '_selected'].includes(column.key)
                    )}
                    items={displayedItems}
                    compact={true}
                    selectionMode={0}
                    selectionPreservedOnEmptyClick={true}
                    onRenderRow={(props): any => {
                        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                        return <ExpandableDetails detailsProps={props!} isExpand={props!.item._expandDetails} />;
                    }}
                />
            )}
            {intermediateDialogTrial.length !== 0 && (
                // {intermediateDialogTrial !== undefined && (
                <Compare
                    title='Intermediate results'
                    trials={intermediateDialogTrial}
                    onHideDialog={(): void => {
                        setIntermediateDialogTrial([]);
                    }}
                />
            )}
            {customizeColumnsDialogVisible && (
                <ChangeColumnComponent
                    selectedColumns={displayedColumns}
                    allColumns={columns
                        .filter(column => !column.key.startsWith('_'))
                        .map(column => ({ key: column.key, name: column.name }))}
                    onSelectedChange={_updateDisplayedColumns}
                    onHideDialog={(): void => {
                        setCustomizeColumnsDialogVisible(false);
                    }}
                    whichComponent='table'
                />
            )}
            {/* Clone a trial and customize a set of new parameters */}
            {/* visible is done inside because prompt is needed even when the dialog is closed */}
            <Customize
                visible={copiedTrialId !== undefined}
                copyTrialId={copiedTrialId || ''}
                closeCustomizeModal={(): void => {
                    setCopiedTrialId(undefined);
                }}
            />
        </div>
    );
};

export default TableList;
