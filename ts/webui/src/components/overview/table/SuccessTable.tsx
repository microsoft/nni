import * as React from 'react';
import {
    DetailsList,
    IDetailsListProps,
    IColumn,
    IRenderFunction,
    IDetailsHeaderProps,
    Sticky,
    StickyPositionType,
    ScrollablePane,
    ScrollbarVisibility
} from '@fluentui/react';
import DefaultMetric from '../../public-child/DefaultMetric';
import Details from './Details';
import { convertDuration } from '../../../static/function';
import { TRIALS } from '../../../static/datamodel';
import { DETAILTABS } from '../../stateless-component/NNItabs';
import '../../../static/style/succTable.scss';
import '../../../static/style/tableStatus.css';
import '../../../static/style/openRow.scss';

interface SuccessTableProps {
    trialIds: string[];
}

interface SuccessTableState {
    columns: IColumn[];
    source: Array<any>;
    innerWidth: number;
}

class SuccessTable extends React.Component<SuccessTableProps, SuccessTableState> {
    constructor(props: SuccessTableProps) {
        super(props);
        this.state = {
            columns: this.columns,
            source: TRIALS.table(this.props.trialIds),
            innerWidth: window.innerWidth
        };
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
        <React.Fragment>
            The experiment is running, please wait for the final metric patiently. You could also find status of trial
            job with <span>{DETAILTABS}</span> button.
        </React.Fragment>
    );

    columns = [
        {
            name: 'Trial No.',
            key: 'sequenceId',
            fieldName: 'sequenceId', // required!
            minWidth: 65,
            maxWidth: 119,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.sequenceId}</div>
        },
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 65,
            maxWidth: 119,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.id}</div>
        },
        {
            name: 'Duration',
            key: 'duration',
            minWidth: 90,
            maxWidth: 166,
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
            minWidth: 108,
            maxWidth: 160,
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
            minWidth: 108,
            maxWidth: 166,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <DefaultMetric trialId={item.id} />
        }
    ];

    onRenderDetailsHeader: IRenderFunction<IDetailsHeaderProps> = (props, defaultRender) => {
        if (!props) {
            return null;
        }
        return (
            <Sticky stickyPosition={StickyPositionType.Header} isScrollSynced>
                {// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                defaultRender!({
                    ...props
                })}
            </Sticky>
        );
    };

    setInnerWidth = (): void => {
        this.setState(() => ({ innerWidth: window.innerWidth }));
    };

    componentDidMount(): void {
        window.addEventListener('resize', this.setInnerWidth);
    }
    componentWillUnmount(): void {
        window.removeEventListener('resize', this.setInnerWidth);
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

        return (
            <div id='succTable'>
                <ScrollablePane className='scrollPanel' scrollbarVisibility={ScrollbarVisibility.auto}>
                    <DetailsList
                        columns={columns}
                        items={source}
                        setKey='set'
                        compact={true}
                        onRenderRow={this.onRenderRow}
                        onRenderDetailsHeader={this.onRenderDetailsHeader}
                        selectionMode={0} // close selector function
                        className='succTable'
                    />
                </ScrollablePane>
                {isNoneData && <div className='succTable-tooltip'>{this.tooltipStr}</div>}
            </div>
        );
    }
}

export default SuccessTable;
