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
import { formatTimeStyle, copyAndSort } from '@static/function';
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
        The experiment is running, please wait for the final metric patiently. You could also find status of trial job
        with <span>{DETAILTABS}</span> button.
    </React.Fragment>
);

const SuccessTable = (props: SuccessTableProps): any => {
    const { trialIds, expandRowIDs, updateOverviewPage, changeExpandRowIDs } = props;
    const [source, setSource] = useState(TRIALS.table(trialIds) as Array<any>);
    const [sortInfo, setSortInfo] = useState({ field: '', isDescend: false } as SortInfo);

    const expandTrialId = (_event: any, id: string): void => {
        changeExpandRowIDs(id);
        updateOverviewPage();
    };
    const onColumnClick = (_ev: React.MouseEvent<HTMLElement>, getColumn: IColumn): void => {
        // eslint-disable-next-line @typescript-eslint/no-use-before-define
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
        // eslint-disable-next-line @typescript-eslint/no-use-before-define
        setColumns(newColumns);
        setSource(newItems);
        setSortInfo({ field: currColumn.fieldName!, isDescend: currColumn.isSortedDescending });
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
                    className='cursor bold positionTopSuccess'
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
            minWidth: 133,
            maxWidth: 255,
            isResizable: true,
            data: 'number',
            onColumnClick: onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='No succeed-padding'>{item.sequenceId}</div>
        },
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 160,
            maxWidth: 330,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <Stack horizontal className='idCopy'>
                    <div className='succeed-padding id'>{item.id}</div>
                    <CopyButton value={item.id} />
                </Stack>
            )
        },
        {
            name: 'Duration',
            key: 'duration',
            minWidth: 124,
            maxWidth: 300,
            isResizable: true,
            fieldName: 'duration',
            data: 'number',
            onColumnClick: onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <div className='duration-global-color duration-list succeed-padding'>
                    <div dangerouslySetInnerHTML={{ __html: formatTimeStyle(item.duration) }} />
                </div>
            )
        },
        {
            name: 'Status',
            key: 'status',
            minWidth: 152,
            maxWidth: 288,
            isResizable: true,
            fieldName: 'status',
            onRender: (item: any): React.ReactNode => (
                <span className={`${item.status} size16 succeed-padding`}>{item.status}</span>
            )
        },
        {
            name: 'Default metric',
            key: 'accuracy',
            fieldName: 'accuracy',
            minWidth: 132,
            maxWidth: 200,
            isResizable: true,
            data: 'number',
            onColumnClick: onColumnClick,
            onRender: (item: any): React.ReactNode => <DefaultMetric trialId={item.id} />
        }
    ];
    const [columns, setColumns] = useState(successTableColumns as IColumn[]);
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

    const onRenderRow: IDetailsListProps['onRenderRow'] = rows => {
        if (rows) {
            return (
                <div>
                    <div>
                        <DetailsRow {...rows} />
                    </div>
                    {Array.from(expandRowIDs).map(
                        item => item === rows.item.id && <OpenRow key={item} trialId={item} />
                    )}
                </div>
            );
        }
        return null;
    };

    useEffect(() => {
        setSource(TRIALS.table(trialIds));
    }, [trialIds]);

    return (
        <div id='succTable'>
            <ScrollablePane className='scrollPanel' scrollbarVisibility={ScrollbarVisibility.auto}>
                <DetailsList
                    columns={columns}
                    items={keepSortedSource}
                    setKey='set'
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
