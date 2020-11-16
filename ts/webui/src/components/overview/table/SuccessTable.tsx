import React from 'react';
import { DetailsList, IDetailsListProps, IColumn, Icon, DetailsRow } from '@fluentui/react';
import DefaultMetric from '../../public-child/DefaultMetric';
import OpenRow from '../../public-child/OpenRow';
import { convertDuration } from '../../../static/function';
import { TRIALS } from '../../../static/datamodel';
import { DETAILTABS } from '../../stateless-component/NNItabs';
import '../../../static/style/succTable.scss';
import '../../../static/style/tableStatus.css';
import '../../../static/style/openRow.scss';

interface SuccessTableProps {
    trialIds: string[];
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    updateOverviewPage: () => void;
}

interface SuccessTableState {
    columns: IColumn[];
    source: Array<any>;
    expandRowIdList: Set<string>;
}

class SuccessTable extends React.Component<SuccessTableProps, SuccessTableState> {
    constructor(props: SuccessTableProps) {
        super(props);
        this.state = {
            columns: this.columns,
            source: TRIALS.table(this.props.trialIds),
            expandRowIdList: new Set() // store expanded row's trial id
        };
    }

    componentDidUpdate(prevProps: SuccessTableProps): void {
        if (this.props.trialIds !== prevProps.trialIds) {
            const { trialIds } = this.props;
            this.setState(() => ({ source: TRIALS.table(trialIds) }));
        }
    }

    render(): React.ReactNode {
        const { columns, source } = this.state;
        const isNoneData = source.length === 0 ? true : false;
        console.info(source);
        return (
            <div id='succTable'>
                <DetailsList
                    columns={columns}
                    items={source}
                    setKey='set'
                    compact={true}
                    onRenderRow={this.onRenderRow}
                    selectionMode={0} // close selector function
                    className='succTable'
                />
                {isNoneData && <div className='succTable-tooltip'>{this.tooltipStr}</div>}
            </div>
        );
    }

    private onColumnClick = (ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
        const { columns, source } = this.state;
        const newColumns: IColumn[] = columns.slice();
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
        const newItems = this.copyAndSort(source, currColumn.fieldName!, currColumn.isSortedDescending);
        this.setState({
            columns: newColumns,
            source: newItems
        });
    };

    private tooltipStr = (
        <React.Fragment>
            The experiment is running, please wait for the final metric patiently. You could also find status of trial
            job with <span>{DETAILTABS}</span> button.
        </React.Fragment>
    );
    
    columns = [
        {
            key: '_expand',
            name: '',
            onRender: (item: any): any => (
                <Icon
                    aria-hidden={true}
                    iconName='ChevronRight'
                    styles={{
                        root: {
                            transition: 'all 0.2s',
                            transform: `rotate(${this.state.expandRowIdList.has(item.id) ? 90 : 0}deg)`
                        }
                    }}
                    onClick={this.expandTrialId.bind(this, Event, item.id)}
                />
            ),
            fieldName: 'expand',
            isResizable: false,
            minWidth: 20,
            maxWidth: 20
        },
        {
            name: 'Trial No.',
            key: 'sequenceId',
            fieldName: 'sequenceId', // required!
            minWidth: 50,
            maxWidth: 87,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.sequenceId}</div>
        },
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 50,
            maxWidth: 87,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.id}</div>
        },
        {
            name: 'Duration',
            key: 'duration',
            minWidth: 65,
            maxWidth: 150,
            isResizable: true,
            fieldName: 'duration',
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <div className='durationsty succeed-padding'>
                    <div>{convertDuration(item.duration)}</div>
                </div>
            )
        },
        {
            name: 'Status',
            key: 'status',
            minWidth: 80,
            maxWidth: 150,
            isResizable: true,
            fieldName: 'status',
            onRender: (item: any): React.ReactNode => (
                <div className={`${item.status} commonStyle succeed-padding`}>{item.status}</div>
            )
        },
        {
            name: 'Default metric',
            key: 'accuracy',
            fieldName: 'accuracy',
            minWidth: 100,
            maxWidth: 160,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <DefaultMetric trialId={item.id} />
        }
    ];

    private onRenderRow: IDetailsListProps['onRenderRow'] = props => {
        const { expandRowIdList } = this.state;
        if (props) {
            return (
                <div>
                    <div>
                        <DetailsRow {...props} />
                    </div>
                    {Array.from(expandRowIdList).map(
                        item => item === props.item.id && <OpenRow key={item} trialId={item} />
                    )}
                </div>
            );
        }
        return null;
    };

    private copyAndSort<T>(items: T[], columnKey: string, isSortedDescending?: boolean): T[] {
        const key = columnKey as keyof T;
        return items.slice(0).sort((a: T, b: T) => ((isSortedDescending ? a[key] < b[key] : a[key] > b[key]) ? 1 : -1));
    }

    private expandTrialId = (_event: any, id: string): void => {
        const { expandRowIdList } = this.state;
        const { updateOverviewPage } = this.props;
        const a = expandRowIdList;
        if (a.has(id)) {
            a.delete(id);
        } else {
            a.add(id);
        }
        this.setState(() => ({ expandRowIdList: a }));
        updateOverviewPage();
    };
}

export default SuccessTable;
