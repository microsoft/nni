import * as React from 'react';
import {
    DetailsList,
    IDetailsListProps,
    IColumn,
    Icon,
    DetailsRow,
    IRenderFunction,
    IDetailsHeaderProps,
    Sticky,
    StickyPositionType,
    ScrollablePane,
    ScrollbarVisibility
} from '@fluentui/react';
import DefaultMetric from '../../public-child/DefaultMetric';
import OpenRow from '../../public-child/OpenRow';
import { convertDuration, copyAndSort } from '../../../static/function';
import { TRIALS } from '../../../static/datamodel';
import { SortInfo } from '../../../static/interface';
import { DETAILTABS } from '../../stateless-component/NNItabs';
import '../../../static/style/succTable.scss';
import '../../../static/style/tableStatus.css';
import '../../../static/style/openRow.scss';

interface SuccessTableProps {
    trialIds: string[];
    updateOverviewPage: () => void;
    expandRowIDs: Set<string>;
    changeExpandRowIDs: Function;
}

interface SuccessTableState {
    columns: IColumn[];
    source: Array<any>;
    sortInfo: SortInfo;
}

class SuccessTable extends React.Component<SuccessTableProps, SuccessTableState> {
    constructor(props: SuccessTableProps) {
        super(props);
        this.state = {
            columns: this.columns,
            source: TRIALS.table(this.props.trialIds),
            sortInfo: { field: '', isDescend: false }
        };
    }

    componentDidUpdate(prevProps: SuccessTableProps): void {
        if (this.props.trialIds !== prevProps.trialIds) {
            const { trialIds } = this.props;
            this.setState(() => ({ source: TRIALS.table(trialIds) }));
        }
    }

    render(): React.ReactNode {
        const { columns, source, sortInfo } = this.state;
        const keepSortedSource = copyAndSort(source, sortInfo.field, sortInfo.isDescend);
        const isNoneData = source.length === 0 ? true : false;

        return (
            <div id='succTable'>
                <ScrollablePane className='scrollPanel' scrollbarVisibility={ScrollbarVisibility.auto}>
                    <DetailsList
                        columns={columns}
                        items={keepSortedSource}
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

    private onColumnClick = (_ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
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
        const newItems = copyAndSort(source, currColumn.fieldName!, currColumn.isSortedDescending);
        this.setState({
            columns: newColumns,
            source: newItems,
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            sortInfo: { field: currColumn.fieldName!, isDescend: currColumn.isSortedDescending }
        });
    };

    private tooltipStr = (
        <React.Fragment>
            The experiment is running, please wait for the final metric patiently. You could also find status of trial
            job with <span>{DETAILTABS}</span> button.
        </React.Fragment>
    );

    private columns: IColumn[] = [
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
                            transform: `rotate(${this.props.expandRowIDs.has(item.id) ? 90 : 0}deg)`
                        }
                    }}
                    className='cursor'
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
            minWidth: 60,
            maxWidth: 100,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.sequenceId}</div>
        },
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 60,
            maxWidth: 90,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.id}</div>
        },
        {
            name: 'Duration',
            key: 'duration',
            minWidth: 80,
            maxWidth: 120,
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
            minWidth: 88,
            maxWidth: 120,
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
            maxWidth: 166,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <DefaultMetric trialId={item.id} />
        }
    ];

    private onRenderDetailsHeader: IRenderFunction<IDetailsHeaderProps> = (props, defaultRender) => {
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

    private onRenderRow: IDetailsListProps['onRenderRow'] = props => {
        const { expandRowIDs } = this.props;
        if (props) {
            return (
                <div>
                    <div>
                        <DetailsRow {...props} />
                    </div>
                    {Array.from(expandRowIDs).map(
                        item => item === props.item.id && <OpenRow key={item} trialId={item} />
                    )}
                </div>
            );
        }
        return null;
    };

    private expandTrialId = (_event: any, id: string): void => {
        const { updateOverviewPage, changeExpandRowIDs } = this.props;
        changeExpandRowIDs(id);
        updateOverviewPage();
    };
}

export default SuccessTable;
