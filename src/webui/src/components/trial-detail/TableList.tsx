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
    SelectionMode,
    IStackTokens
} from '@fluentui/react';
import ReactPaginate from 'react-paginate';
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

const defaultDisplayedColumns = ['sequenceId', 'id', 'startTime', 'endTime', 'duration', 'status', 'metrics/default'];

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

const horizontalGapStackTokens: IStackTokens = {
    childrenGap: 20,
    padding: 10
};

interface TableListProps {
    tableSource: TableObj[];
    trialsUpdateBroadcast: number;
}

interface SortInfo {
    field: string;
    isDescend?: boolean;
}

interface TableListState {
    displayedItems: any[];
    displayedColumns: string[];
    columns: IColumn[];
    searchType: SearchOptionType;
    searchText: string;
    tablePageSize: number;
    customizeColumnsDialogVisible: boolean;
}

class TableList extends React.Component<TableListProps, TableListState> {
    constructor(props: TableListProps) {
        super(props);

        this.state = {
            displayedItems: [],
            displayedColumns: defaultDisplayedColumns,
            columns: [],
            searchType: 'id',
            searchText: '',
            tablePageSize: 20,
            customizeColumnsDialogVisible: false
        };
    }

    private _onTablePageSizeSelect = (
        event: React.FormEvent<HTMLDivElement>,
        item: IDropdownOption | undefined
    ): void => {
        if (item !== undefined) {
            this.setState({ tablePageSize: item.text === 'all' ? -1 : parseInt(item.text, 10) });
        }
    };

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
                startTime: formatTimestamp(trial.startTime, '--'),
                endTime: formatTimestamp(trial.endTime, '--'),
                duration: convertDuration(trial.duration),
                status: trial.status,
                intermediateCount: trial.intermediates.length,
                _expandDetails: false // hidden field names should start with `_`
            };
            for (const [k, v] of trial.parameters(searchSpace)) {
                ret[`space/${k.baseName}`] = v;
            }
            for (const [k, v] of trial.metrics(metricSpace)) {
                ret[`metrics/${k.baseName}`] = v;
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
        for (const [k, v] of Object.entries(tableItems[0])) {
            // TODO: add blacklist
            columns.push({
                name: k,
                key: k,
                fieldName: k,
                minWidth: 150,
                maxWidth: 400,
                isResizable: true,
                onColumnClick: this._onColumnClick.bind(this)
            });
        }
        return columns;
    }

    private _showCompareDialog(): void {
        console.log('Compare button clicked'); // eslint-disable-line no-console
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

    render(): React.ReactNode {
        const perPageOptions = [
            { key: '10', text: '10 items per page' },
            { key: '20', text: '20 items per page' },
            { key: '50', text: '50 items per page' },
            { key: 'all', text: 'All items' }
        ];
        const { displayedItems, columns, searchType, customizeColumnsDialogVisible, displayedColumns } = this.state;

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
                        <DefaultButton text='Compare' className='allList-compare' onClick={this._showCompareDialog} />
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
                {columns && (
                    <DetailsList
                        columns={columns.filter(column => displayedColumns.includes(column.key))}
                        items={displayedItems}
                        compact={true}
                        selectionMode={SelectionMode.multiple}
                        onRenderRow={props => {
                            return <ExpandableDetails detailsProps={props!} isExpand={props!.item._expandDetails} />;
                        }}
                    />
                )}
                {/* 
                    <Stack horizontal horizontalAlign='end' verticalAlign='baseline' styles={{ root: { padding: 10 } }} tokens={horizontalGapStackTokens}>
                        <Dropdown
                            selectedKey={this.state.perPage === this.props.tableSource.length ? 'all' : String(this.state.perPage)}
                            options={perPageOptions}
                            onChange={this.updatePerPage}
                            styles={{ dropdown: { width: 150 } }} />

                        <ReactPaginate
                            previousLabel={'<'}
                            nextLabel={'>'}
                            breakLabel={'...'}
                            breakClassName={'break'}
                            pageCount={this.state.pageCount}
                            marginPagesDisplayed={2}
                            pageRangeDisplayed={2}
                            onPageChange={this.handlePageClick}
                            containerClassName={(this.props.tableSource.length == 0 ? 'pagination hidden' : 'pagination')}
                            subContainerClassName={'pages pagination'}
                            disableInitialCallback={false}
                            activeClassName={'active'} />

                    </Stack> */}
                <ChangeColumnComponent
                    hidden={!customizeColumnsDialogVisible}
                    selectedColumns={displayedColumns}
                    allColumns={columns
                        .filter(column => !column.key.startsWith('_'))
                        .map(column => ({ key: column.key, name: column.fieldName! }))}
                    onSelectedChange={this._updateDisplayedColumns.bind(this)}
                    onHideDialog={() => {
                        this.setState({ customizeColumnsDialogVisible: false });
                    }}
                />
            </div>
        );
    }
}

export default TableList;
