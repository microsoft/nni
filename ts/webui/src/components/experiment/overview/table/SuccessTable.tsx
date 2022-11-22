import React, { useState, useEffect } from 'react';
import {
    Stack,
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
import DefaultMetric from './DefaultMetric';
import OpenRow from '@/components/common/ExpandableDetails/OpenRow';
import CopyButton from '@components/common/CopyButton';
import { convertDuration, copyAndSort } from '@static/function';
import { TRIALS } from '@static/datamodel';
import { SortInfo } from '@static/interface';
import { DETAILTABS } from '@components/nav/slideNav/NNItabs';
import '@style/experiment/overview/succTable.scss';
import '@style/common/trialStatus.css';
import '@style/openRow.scss';

interface SuccessTableProps {
    trialIds: string[];
    updateOverviewPage: () => void;
    expandRowIDs: Set<string>;
    changeExpandRowIDs: Function;
}

const tooltipStr = (
    <React.Fragment>
        The experiment is running, please wait for the final metric patiently. You could also find status of trial
        job with <span>{DETAILTABS}</span> button.
    </React.Fragment>
);

const SuccessTable = (props: SuccessTableProps): any => {
    const onColumnClick = (_ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
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
        setColumns(newColumns);
        setSource(newItems);
        setSortInfo( { field: currColumn.fieldName!, isDescend: currColumn.isSortedDescending });
    };
    
    const successTableColumns: IColumn[] = [
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
                            transform: `rotate(${expandRowIDs.has(item.id) ? 90 : 0}deg)`
                        }
                    }}
                    className='cursor bold positionTop'
                    onClick={expandTrialId.bind(this, Event, item.id)}
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
            maxWidth: 80,
            isResizable: true,
            data: 'number',
            onColumnClick: onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.sequenceId}</div>
        },
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 90,
            maxWidth: 100,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <Stack horizontal className='idCopy'>
                    <div className='succeed-padding'>{item.id}</div>
                    <CopyButton value={item.id} />
                </Stack>
            )
        },
        {
            name: 'Duration',
            key: 'duration',
            minWidth: 70,
            maxWidth: 120,
            isResizable: true,
            fieldName: 'duration',
            data: 'number',
            onColumnClick: onColumnClick,
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
            onColumnClick: onColumnClick,
            onRender: (item: any): React.ReactNode => <DefaultMetric trialId={item.id} />
        }
    ];
    const { trialIds, expandRowIDs, updateOverviewPage, changeExpandRowIDs } = props;
    const [columns, setColumns] = useState(successTableColumns as IColumn[]);
    // 这个不应该是准确的吗？怎么是any呢
    const [source, setSource] = useState(TRIALS.table(trialIds) as Array<any>);
    const [sortInfo, setSortInfo] = useState({ field: '', isDescend: false } as SortInfo);
    useEffect(() => {
        setSource(TRIALS.table(trialIds));
    }, [trialIds]);

    const keepSortedSource = copyAndSort(source, sortInfo.field, sortInfo.isDescend);
    const isNoneData = source.length === 0 ? true : false;

    const onRenderDetailsHeader: IRenderFunction<IDetailsHeaderProps> = (props, defaultRender) => {
        if (!props) {
            return null;
        }
        return (
            <Sticky stickyPosition={StickyPositionType.Header} isScrollSynced>
                {
                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    defaultRender!({
                        ...props
                    })
                }
            </Sticky>
        );
    };

    const onRenderRow: IDetailsListProps['onRenderRow'] = props => {
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

    const expandTrialId = (_event: any, id: string): void => {
        changeExpandRowIDs(id);
        updateOverviewPage();
    };

    return (
        <div id='succTable'>
            <ScrollablePane className='scrollPanel' scrollbarVisibility={ScrollbarVisibility.auto}>
                <DetailsList
                    columns={columns}
                    items={keepSortedSource}
                    setKey='set'
                    compact={true}
                    onRenderRow={onRenderRow}
                    onRenderDetailsHeader={onRenderDetailsHeader}
                    selectionMode={0} // close selector function
                    className='succTable'
                />
            </ScrollablePane>
            {isNoneData && <div className='succTable-tooltip'>{tooltipStr}</div>}
        </div>
    );
};

export default SuccessTable;
