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
    displayedTrials: TableObj[];
    expandedIds: string[];
}

class TableList extends React.Component<TableListProps, TableListState> {

    constructor(props: TableListProps) {
        super(props);

        this.state = {
            displayedTrials: props.tableSource,  // TODO: pagination and sorting
            expandedIds: []
        };
    }

    // // sort for table column
    // onColumnClick = (ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
    // };

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

    private trialsToTableItems(trials: TableObj[]): any[] {
        // FIXME use search space and metrics space from TRIALS will cause update issues.
        const searchSpace = TRIALS.inferredSearchSpace(EXPERIMENT.searchSpaceNew);
        const metricSpace = TRIALS.inferredMetricSpace();
        return trials.map((trial) => {
            const ret = {
                sequenceId: trial.sequenceId,
                id: trial.id,
                startTime: formatTimestamp(trial.startTime),
                endTime: formatTimestamp(trial.endTime, '--'),
                duration: convertDuration(trial.duration),
                status: trial.status,
                intermediateCount: trial.intermediates.length
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

    private buildColumnsFromTableItems(tableItems: any[]): IColumn[] {
        const columns: IColumn[] = [{
            key: 'expand',
            name: '',
            onRender: (item, index, column) => {
                return <Icon aria-hidden={true} iconName="ChevronDown"
                    onClick={(event) => {
                        event.preventDefault();
                        this.setState({
                            expandedIds: [...this.state.expandedIds, item.id]
                        });
                    }} />
            },
            fieldName: 'expand',
            isResizable: false,
            minWidth: 30,
            maxWidth: 30
        }];
        // looking at the first row only for now
        for (const [k, v] of Object.entries(tableItems[0])) {
            columns.push({
                name: k,
                key: k,
                fieldName: k,
                minWidth: 150,
                maxWidth: 400,
                isResizable: true,
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

    render(): React.ReactNode {
        const perPageOptions = [
            { key: '10', text: '10 items per page' },
            { key: '20', text: '20 items per page' },
            { key: '50', text: '50 items per page' },
            { key: 'all', text: 'All items' },
        ];
        const { tableSource } = this.props;
        const { expandedIds } = this.state;

        const items = this.trialsToTableItems(tableSource);
        const columns = this.buildColumnsFromTableItems(items);

        console.log(items);  // eslint-disable-line no-console
        console.log(columns);  // eslint-disable-line no-console

        return (
            <Stack>
                <div id="tableList">
                    <DetailsList
                        columns={columns}
                        items={items}
                        compact={true}
                        selectionMode={SelectionMode.multiple}
                        onRenderRow={(props) => {
                            return <ExpandableDetails detailsProps={props!}
                                isExpand={expandedIds.includes(props!.item.id)} />;
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
