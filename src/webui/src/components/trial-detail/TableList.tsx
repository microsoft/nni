import React, { lazy } from 'react';
import {
    Stack,
    Dropdown,
    DetailsList,
    Icon,
    StackItem,
    DefaultButton,
    IDropdownOption,
    IColumn,
    Selection,
    SelectionMode,
    IStackTokens
} from '@fluentui/react';
import { LineChart, blocked, copy, tableListIcon } from '../buttons/Icon';
import { MANAGER_IP, COLUMNPro } from '../../static/const';
import { convertDuration, formatTimestamp, intermediateGraphOption, parseMetrics } from '../../static/function';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { SearchSpace, TableObj, TrialJobInfo, MultipleAxes } from '../../static/interface';
import ExpandableDetails from '../public-child/ExpandableDetails';
import ChangeColumnComponent from '../modals/ChangeColumnComponent';
import '../../static/style/search.scss';
import '../../static/style/tableStatus.css';
import '../../static/style/logPath.scss';
import '../../static/style/table.scss';
import '../../static/style/button.scss';
import '../../static/style/openRow.scss';
import '../../static/style/pagination.scss';
import { TrialManager } from '../../static/model/trialmanager';
import PaginationTable from '../public-child/PaginationTable';
import Compare from '../modals/Compare';

const echarts = require('echarts/lib/echarts');
require('echarts/lib/chart/line');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
echarts.registerTheme('my_theme', {
    color: '#3c8dbc'
});

type SearchOptionType = 'id' | 'trialnum' | 'status' | 'parameters';
const searchOptionLiterals = {
    id: 'ID',
    trialnum: 'Trial No.',
    status: 'Status',
    parameters: 'Parameters'
};

const defaultDisplayedColumns = ['sequenceId', 'id', 'startTime', 'endTime', 'duration', 'status', 'metric/default'];

function _copyAndSort<T>(items: T[], columnKey: string, isSortedDescending?: boolean): any {
    const key = columnKey as keyof T;
    return items.slice(0).sort(function(a: T, b: T): any {
        if (a[key] === undefined) {
            return 1;
        }
        if (b[key] === undefined) {
            return -1;
        }
        return (isSortedDescending ? a[key] < b[key] : a[key] > b[key]) ? 1 : -1;
    });
}

function _inferColumnTitle(columnKey: string): string {
    if (columnKey === 'sequenceId') {
        return 'Trial No.';
    } else if (columnKey === 'id') {
        return 'ID';
    } else if (columnKey === 'intermediateCount') {
        return 'Intermediate results (#)';
    } else if (columnKey.startsWith('space/')) {
        return 'Space: ' + columnKey.split('/', 2)[1];
    } else if (columnKey.startsWith('metric/')) {
        return 'Metric: ' + columnKey.split('/', 2)[1];
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
}

class TableList extends React.Component<TableListProps, TableListState> {
    private _selection: Selection;

    constructor(props: TableListProps) {
        super(props);

        this.state = {
            displayedItems: [],
            displayedColumns: defaultDisplayedColumns,
            columns: [],
            searchType: 'id',
            searchText: '',
            customizeColumnsDialogVisible: false,
            compareDialogVisible: false,
            selectedRowIds: []
        };

        this._selection = new Selection({
            onSelectionChanged: () => {
                console.log(this._selection.getSelection().map(s => (s as any).id)); // eslint-disable-line no-console
                // this.setState({
                //     selectedRowIds: []
                // });
                this.setState({
                    selectedRowIds: this._selection.getSelection().map(s => (s as any).id)
                });
            }
        });
    }

    /* Search related methods */

    // This functions as the filter for the final trials displayed in the current table
    private _filterTrials(trials: TableObj[]): TableObj[] {
        const { searchText, searchType } = this.state;
        // search a trial by Trial No. | Trial ID | Parameters | Status
        let searchFilter = (_: TableObj): boolean => true;
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
        const { columns, displayedItems } = this.state;
        const newColumns: IColumn[] = columns.slice();
        const currColumn: IColumn = newColumns.filter(currCol => column.key === currCol.key)[0];
        newColumns.forEach((newCol: IColumn) => {
            if (newCol === currColumn) {
                currColumn.isSortedDescending = !currColumn.isSortedDescending;
                currColumn.isSorted = true;
            } else {
                newCol.isSorted = false;
                newCol.isSortedDescending = true;
            }
        });
        const newItems = _copyAndSort(displayedItems, currColumn.fieldName!, currColumn.isSortedDescending);
        this.setState({
            columns: newColumns,
            displayedItems: newItems
        });
    }

    private _trialsToTableItems(trials: TableObj[]): any[] {
        // TODO: use search space and metrics space from TRIALS will cause update issues.
        const searchSpace = TRIALS.inferredSearchSpace(EXPERIMENT.searchSpaceNew);
        const metricSpace = TRIALS.inferredMetricSpace();
        return trials.map(trial => {
            const ret = {
                sequenceId: trial.sequenceId,
                id: trial.id,
                startTime: formatTimestamp((trial as any).info.startTime, '--'), // FIXME: why do we need info here?
                endTime: formatTimestamp((trial as any).info.endTime, '--'),
                duration: convertDuration(trial.duration),
                status: trial.status,
                intermediateCount: trial.intermediates.length,
                _expandDetails: false // hidden field names should start with `_`
            };
            for (const [k, v] of trial.parameters(searchSpace)) {
                ret[`space/${k.baseName}`] = v;
            }
            for (const [k, v] of trial.metrics(metricSpace)) {
                ret[`metric/${k.baseName}`] = v;
            }
            return ret;
        });
    }

    private _buildColumnsFromTableItems(tableItems: any[]): IColumn[] {
        // extra column, for a icon to expand the trial details panel
        const columns: IColumn[] = [
            {
                key: '_expand',
                name: '',
                onRender: (item, index) => {
                    return (
                        <Icon
                            aria-hidden={true}
                            iconName='ChevronRight'
                            styles={{
                                root: {
                                    transition: 'all 0.2s',
                                    transform: `rotate(${item._expandDetails ? 90 : 0}deg)`
                                }
                            }}
                            onClick={event => {
                                event.stopPropagation();
                                const newItem: any = { ...item, _expandDetails: !item._expandDetails };
                                const newItems = [...this.state.displayedItems];
                                newItems[index!] = newItem;
                                this.setState({
                                    displayedItems: newItems
                                });
                            }}
                            onMouseDown={e => {
                                e.stopPropagation();
                            }}
                            onMouseUp={e => {
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
            const lengths = tableItems.map(item => `${item[k]}`.length);
            const avgLengths = lengths.reduce((a, b) => a + b) / lengths.length;
            const columnTitle = _inferColumnTitle(k);
            const columnWidth = Math.max(columnTitle.length, avgLengths);
            // TODO: add blacklist
            columns.push({
                name: columnTitle,
                key: k,
                fieldName: k,
                minWidth: columnWidth * 4,
                maxWidth: columnWidth * 10,
                isResizable: true,
                onColumnClick: this._onColumnClick.bind(this)
            });
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

    componentDidUpdate(prevProps: TableListProps): void {
        if (this.props.tableSource !== prevProps.tableSource) {
            this._updateTableSource();
        }
    }

    componentDidMount(): void {
        this._updateTableSource();
    }

    render(): React.ReactNode {
        const {
            displayedItems,
            columns,
            searchType,
            customizeColumnsDialogVisible,
            compareDialogVisible,
            displayedColumns,
            selectedRowIds
        } = this.state;

        console.log(displayedItems); // eslint-disable-line no-console
        console.log(columns); // eslint-disable-line no-console

        return (
            <div id='tableList'>
                <Stack horizontal className='panelTitle' style={{ marginTop: 10 }}>
                    <span style={{ marginRight: 12 }}>{tableListIcon}</span>
                    <span>Trial jobs</span>
                </Stack>
                <Stack horizontal className='allList'>
                    <StackItem grow={50}>
                        {selectedRowIds.length > 0 && (
                            <DefaultButton
                                text='Compare'
                                className='allList-compare'
                                onClick={() => {
                                    this.setState({ compareDialogVisible: true });
                                }}
                            />
                        )}
                    </StackItem>
                    <StackItem grow={50}>
                        <Stack horizontal horizontalAlign='end' className='allList'>
                            <DefaultButton
                                className='allList-button-gap'
                                text='Customize columns'
                                onClick={() => {
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
                                placeholder={`Search by ${searchOptionLiterals[searchType]}`}
                                onChange={this._updateSearchText.bind(this)}
                                style={{ width: 230 }}
                            />
                        </Stack>
                    </StackItem>
                </Stack>
                {columns && displayedItems && (
                    <PaginationTable
                        columns={columns.filter(
                            column => displayedColumns.includes(column.key) || column.key === '_expand'
                        )}
                        items={displayedItems}
                        compact={true}
                        selection={this._selection}
                        selectionMode={SelectionMode.multiple}
                        selectionPreservedOnEmptyClick={true}
                        onRenderRow={props => {
                            return <ExpandableDetails detailsProps={props!} isExpand={props!.item._expandDetails} />;
                        }}
                    />
                )}
                {compareDialogVisible && (
                    <Compare
                        trials={this.props.tableSource.filter(trial => selectedRowIds.includes(trial.id))}
                        onHideDialog={() => {
                            this.setState({ compareDialogVisible: false });
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
                        onHideDialog={() => {
                            this.setState({ customizeColumnsDialogVisible: false });
                        }}
                    />
                )}
            </div>
        );
    }
}

export default TableList;
