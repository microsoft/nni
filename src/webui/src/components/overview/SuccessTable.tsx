import * as React from 'react';
import { DetailsList, IDetailsListProps, IColumn } from 'office-ui-fabric-react';
import DefaultMetric from '../public-child/DefaultMetric';
import Details from './Details';
import { convertDuration } from '../../static/function';
import { TRIALS } from '../../static/datamodel';
// import '../../static/style/succTable.scss';
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

    private _onRenderRow: IDetailsListProps['onRenderRow'] = props => {
        if (props) {
            return <Details detailsProps={props} />;
        }
        return null;
    };

    _onColumnClick = (ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
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
        const newItems = this._copyAndSort(source, currColumn.fieldName!, currColumn.isSortedDescending);
        this.setState({
            columns: newColumns,
            source: newItems
        });
    };

    private _copyAndSort<T>(items: T[], columnKey: string, isSortedDescending?: boolean): T[] {
        const key = columnKey as keyof T;
        return items.slice(0).sort((a: T, b: T) => ((isSortedDescending ? a[key] < b[key] : a[key] > b[key]) ? 1 : -1));
    }

    columns = [
        {
            name: 'Trial No.',
            key: 'sequenceId',
            fieldName: 'sequenceId', // required!
            minWidth: 80,
            maxWidth: 80,
            data: 'number',
            onColumnClick: this._onColumnClick
        }, {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 80,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: this._onColumnClick
        }, {
            name: 'Duration',
            key: 'duration',
            minWidth: 140,
            fieldName: 'duration',
            data: 'number',
            onColumnClick: this._onColumnClick,
            onRender: (item: any): React.ReactNode => {
                return (
                    <div className="durationsty"><div>{convertDuration(item.duration)}</div></div>
                );
            },
        }, {
            name: 'Status',
            key: 'status',
            minWidth: 150,
            fieldName: 'status',
            onRender: (item: any): React.ReactNode => {
                return (
                    <div className={`${item.status} commonStyle`}>{item.status}</div>
                );
            }
        }, {
            name: 'Default metric',
            key: 'accuracy',
            fieldName: 'accuracy',
            minWidth: 100,
            data: 'number',
            onColumnClick: this._onColumnClick,
            onRender: (item: any): React.ReactNode => {
                return (
                    <DefaultMetric trialId={item.id} />
                );
            }
        }
    ];

    render(): React.ReactNode {
        const { columns, source } = this.state;
        return (
            <div id="succTable">
                {/* TODO: [style] lineHeight question */}
                <DetailsList
                    columns={columns}
                    items={source}
                    setKey="set"
                    onRenderRow={this._onRenderRow}
                />
            </div>
        );
    }
}

export default SuccessTable;
