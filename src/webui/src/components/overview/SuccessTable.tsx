import * as React from 'react';
import { DetailsList, IDetailsListProps, IColumn } from 'office-ui-fabric-react';
import DefaultMetric from '../public-child/DefaultMetric';
import Details from './Details';
import { convertDuration } from '../../static/function';
import { TRIALS } from '../../static/datamodel';
import { DETAILTABS } from '../stateless-component/NNItabs';
import '../../static/style/succTable.scss';
import '../../static/style/openRow.scss';

interface SuccessTableProps {
    trialIds: string[];
}

interface SuccessTableState {
    columns: IColumn[];
    source: Array<any>;
}

class SuccessTable extends React.Component<SuccessTableProps, SuccessTableState> {
    constructor(props: SuccessTableProps) {
        super(props);
        this.state = { columns: this.columns, source: TRIALS.table(this.props.trialIds) };
    }

    private onRenderRow: IDetailsListProps['onRenderRow'] = props => {
        if (props) {
            return <Details detailsProps={props} />;
        }
        return null;
    };

    onColumnClick = (ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
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

    private copyAndSort<T>(items: T[], columnKey: string, isSortedDescending?: boolean): T[] {
        const key = columnKey as keyof T;
        return items.slice(0).sort((a: T, b: T) => ((isSortedDescending ? a[key] < b[key] : a[key] > b[key]) ? 1 : -1));
    }

    tooltipStr = (
        <div>
            <p>The experiment is running, please wait for the final metric patiently.</p>
            <div className='link'>
                You could also find status of trial job with <span>{DETAILTABS}</span> button.
            </div>
        </div>
    );

    columns = [
        {
            name: 'Trial No.',
            key: 'sequenceId',
            fieldName: 'sequenceId', // required!
            minWidth: 60,
            maxWidth: 120,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick
        },
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 80,
            maxWidth: 100,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: this.onColumnClick
        },
        {
            name: 'Duration',
            key: 'duration',
            minWidth: 100,
            maxWidth: 210,
            isResizable: true,
            fieldName: 'duration',
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => {
                return (
                    <div className='durationsty'>
                        <div>{convertDuration(item.duration)}</div>
                    </div>
                );
            }
        },
        {
            name: 'Status',
            key: 'status',
            minWidth: 140,
            maxWidth: 210,
            isResizable: true,
            fieldName: 'status',
            onRender: (item: any): React.ReactNode => {
                return <div className={`${item.status} commonStyle`}>{item.status}</div>;
            }
        },
        {
            name: 'Default metric',
            key: 'accuracy',
            fieldName: 'accuracy',
            minWidth: 120,
            maxWidth: 360,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => {
                return <DefaultMetric trialId={item.id} />;
            }
        }
    ];

    componentDidUpdate(prevProps: SuccessTableProps): void {
        if (this.props.trialIds !== prevProps.trialIds) {
            const { trialIds } = this.props;
            this.setState(() => ({ source: TRIALS.table(trialIds) }));
        }
    }

    render(): React.ReactNode {
        const { columns, source } = this.state;
        const isNoneData = source.length === 0 ? true : false;

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
}

export default SuccessTable;
