import * as React from 'react';
import axios from 'axios';
import ReactEcharts from 'echarts-for-react';
import {
    Stack, Dropdown, DetailsList, IDetailsListProps, DetailsListLayoutMode, Icon,
    PrimaryButton, Modal, IDropdownOption, IColumn, Selection, SelectionMode, IconButton, TooltipHost, IStackTokens
} from 'office-ui-fabric-react';
import ReactPaginate from 'react-paginate';
import { LineChart, blocked, copy } from '../Buttons/Icon';
import { MANAGER_IP, COLUMNPro } from '../../static/const';
import { convertDuration, formatTimestamp, intermediateGraphOption, parseMetrics } from '../../static/function';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { SearchSpace, TableObj, TrialJobInfo, MultipleAxes } from '../../static/interface';
import ExpandableDetails from '../public-child/ExpandableDetails';
import ChangeColumnComponent from '../Modals/ChangeColumnComponent';
import Compare from '../Modals/Compare';
import KillJob from '../Modals/Killjob';
import Customize from '../Modals/CustomizedTrial';
import { contentStyles, iconButtonStyles } from '../Buttons/ModalTheme';
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

function _copyAndSort<T>(items: T[], columnKey: string, isSortedDescending?: boolean): any {
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

const horizontalGapStackTokens: IStackTokens = {
    childrenGap: 20,
    padding: 10,
};

interface TableListProps {
    pageSize: number;
    tableSource: TableObj[];
    trialsUpdateBroadcast: number;
    changeColumn: (val: string[]) => void;
}

interface SortInfo {
    field: string;
    isDescend?: boolean;
}

interface TableListState {
    displayedItems: any[];
    columns: IColumn[];
}

class TableList extends React.Component<TableListProps, TableListState> {

    constructor(props: TableListProps) {
        super(props);

        this.state = {
            displayedItems: [],
            columns: []
        };
    }

    // // sort for table column
    // onColumnClick = (ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
    // };


    private _onColumnClick(ev: React.MouseEvent<HTMLElement>, column: IColumn): void {
        // handle the click events on table header (do sorting)
        console.log(this.state);  // eslint-disable-line no-console
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
            displayedItems: newItems,
        });
    }

    private _trialsToTableItems(trials: TableObj[]): any[] {
        // TODO: use search space and metrics space from TRIALS will cause update issues.
        const searchSpace = TRIALS.inferredSearchSpace(EXPERIMENT.searchSpaceNew);
        const metricSpace = TRIALS.inferredMetricSpace();
        return trials.map((trial) => {
            const ret = {
                sequenceId: trial.sequenceId,
                id: trial.id,
                startTime: formatTimestamp(trial.startTime, '--'),
                endTime: formatTimestamp(trial.endTime, '--'),
                duration: convertDuration(trial.duration),
                status: trial.status,
                intermediateCount: trial.intermediates.length,
                expandDetails: false
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
        const columns: IColumn[] = [{
            key: 'expand',
            name: '',
            onRender: (item, index, column) => {
                return <Icon
                    aria-hidden={true}
                    iconName='ChevronRight'
                    styles={{ root: { transition: 'all 0.2s', transform: `rotate(${item.expandDetails ? 90 : 0}deg)` } }}
                    onClick={(event) => {
                        event.stopPropagation();
                        const newItem: any = { ...item, expandDetails: !item.expandDetails };
                        const newItems = [...this.state.displayedItems];
                        newItems[index!] = newItem;
                        this.setState({
                            displayedItems: newItems
                        });
                    }}
                    onMouseDown={(e) => {
                        e.stopPropagation();
                    }}
                    onMouseUp={(e) => {
                        e.stopPropagation();
                    }} />
            },
            fieldName: 'expand',
            isResizable: false,
            minWidth: 20,
            maxWidth: 20
        }];
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

    compareBtn() {
        console.log('Compare button clicked'); // eslint-disable-line no-console
    }

    customizeColumn() {
        console.log('Customized column clicked'); // eslint-disable-line no-console
    }

    componentDidUpdate(prevProps: TableListProps): void {
        if (this.props.tableSource !== prevProps.tableSource) {
            const items = this._trialsToTableItems(this.props.tableSource);
            const columns = this._buildColumnsFromTableItems(items);
            this.setState({
                displayedItems: items,
                columns: columns
            });
        }
    }

    render(): React.ReactNode {
        const perPageOptions = [
            { key: '10', text: '10 items per page' },
            { key: '20', text: '20 items per page' },
            { key: '50', text: '50 items per page' },
            { key: 'all', text: 'All items' },
        ];
        const { displayedItems, columns } = this.state;

        console.log(displayedItems);  // eslint-disable-line no-console
        console.log(columns);  // eslint-disable-line no-console

        return (
            <Stack>
                <div id="tableList">
                    <DetailsList
                        columns={columns}
                        items={displayedItems}
                        compact={true}
                        selectionMode={SelectionMode.multiple}
                        onRenderRow={(props) => {
                            return <ExpandableDetails detailsProps={props!}
                                isExpand={props!.item.expandDetails} />;
                        }}
                    />
                    {/* 
                    <Stack horizontal horizontalAlign="end" verticalAlign="baseline" styles={{ root: { padding: 10 } }} tokens={horizontalGapStackTokens}>
                        <Dropdown
                            selectedKey={this.state.perPage === this.props.tableSource.length ? 'all' : String(this.state.perPage)}
                            options={perPageOptions}
                            onChange={this.updatePerPage}
                            styles={{ dropdown: { width: 150 } }} />

                        <ReactPaginate
                            previousLabel={"<"}
                            nextLabel={">"}
                            breakLabel={"..."}
                            breakClassName={"break"}
                            pageCount={this.state.pageCount}
                            marginPagesDisplayed={2}
                            pageRangeDisplayed={2}
                            onPageChange={this.handlePageClick}
                            containerClassName={(this.props.tableSource.length == 0 ? "pagination hidden" : "pagination")}
                            subContainerClassName={"pages pagination"}
                            disableInitialCallback={false}
                            activeClassName={"active"} />

                    </Stack> */}

                </div>
            </Stack>
        );
    }
}

export default TableList;
