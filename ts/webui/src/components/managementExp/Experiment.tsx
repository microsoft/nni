import * as React from 'react';
import { Stack, DetailsList, DefaultButton, Icon, SearchBox, IColumn } from '@fluentui/react';
import { ExperimentsManager } from '../../static/model/experimentsManager';
import { formatTimestamp, copyAndSort } from '../../static/function';
import { AllExperimentList, SortInfo } from '../../static/interface';
import { compareDate, filterByStatusOrPlatform, getSortedSource } from './expFunction';
import { Hearder } from './Header';
import NameColumn from './NameColumn';
import FilterBtns from './FilterBtns';
import '../../App.scss';
import '../../static/style/nav/nav.scss';
import '../../static/style/experiment/experiment.scss';
import '../../static/style/common/ellipsis.scss';
import '../../static/style/tableStatus.css';

interface ExpListState {
    columns: IColumn[];
    hideFilter: boolean;
    searchInputVal: string;
    selectedStatus: string;
    selectedPlatform: string;
    selectedStartDate?: Date;
    selectedEndDate?: Date;
    sortInfo: SortInfo;
    source: AllExperimentList[];
    originExperimentList: AllExperimentList[];
    searchSource: AllExperimentList[];
}

class Experiment extends React.Component<{}, ExpListState> {
    constructor(props) {
        super(props);
        this.state = {
            columns: this.columns,
            hideFilter: true,
            searchInputVal: '',
            selectedStatus: '',
            selectedPlatform: '',
            source: [], // data in table
            originExperimentList: [], // api /experiments-info
            searchSource: [], // search box search result
            sortInfo: { field: '', isDescend: false }
        };
    }

    async componentDidMount(): Promise<void> {
        const EXPERIMENTMANAGER = new ExperimentsManager();
        await EXPERIMENTMANAGER.init();
        const result = EXPERIMENTMANAGER.getExperimentList();
        this.setState(() => ({
            source: result,
            originExperimentList: result,
            searchSource: result
        }));
    }

    render(): React.ReactNode {
        const { hideFilter, selectedStatus, source, selectedPlatform, selectedStartDate, selectedEndDate } = this.state;
        return (
            <Stack className='nni' style={{ minHeight: window.innerHeight }}>
                <Hearder />
                <Stack className='contentBox expBackground'>
                    <Stack className='content'>
                        <Stack className='experimentList'>
                            <Stack className='box' horizontal>
                                <div className='search'>
                                    <SearchBox
                                        className='search-input'
                                        placeholder='Search the experiment by name and ID'
                                        onEscape={this.setOriginSource.bind(this)}
                                        onClear={this.setOriginSource.bind(this)}
                                        onChange={this.searchNameAndId.bind(this)}
                                    />
                                </div>
                                <div className='filter'>
                                    <DefaultButton
                                        onClick={this.clickFilter.bind(this)}
                                        className={`${!hideFilter ? 'filter-button-open' : null}`}
                                    >
                                        <Icon iconName='Equalizer' />
                                        <span className='margin'>Filter</span>
                                    </DefaultButton>
                                </div>
                            </Stack>
                            <Stack className={`${hideFilter ? 'hidden' : ''} filter-condition`} horizontal gap={25}>
                                <FilterBtns
                                    selectedStatus={selectedStatus}
                                    selectedPlatform={selectedPlatform}
                                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                                    selectedStartDate={selectedStartDate!}
                                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                                    selectedEndDate={selectedEndDate!}
                                    selectStatus={this.selectStatus.bind(this)}
                                    selectPlatform={this.selectPlatform.bind(this)}
                                    getSelectedData={this.getSelectedData.bind(this)}
                                    setSearchSource={this.setSearchSource.bind(this)}
                                />
                            </Stack>
                            <DetailsList
                                columns={this.columns}
                                items={source}
                                setKey='set'
                                compact={true}
                                selectionMode={0} // close selector function
                                className='table'
                            />
                        </Stack>
                    </Stack>
                </Stack>
            </Stack>
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
        this.setState(() => ({
            columns: newColumns,
            source: newItems,
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            sortInfo: { field: currColumn.fieldName!, isDescend: currColumn.isSortedDescending }
        }));
    };

    private columns: IColumn[] = [
        {
            name: 'Name',
            key: 'experimentName',
            fieldName: 'experimentName', // required!
            minWidth: 100,
            maxWidth: 130,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <NameColumn port={item.port} expName={item.experimentName} />
        },
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: 100,
            maxWidth: 130,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding id'>{item.id}</div>
        },
        {
            name: 'Status',
            key: 'status',
            fieldName: 'status',
            minWidth: 100,
            maxWidth: 150,
            isResizable: true,
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <div className={`${item.status} commonStyle succeed-padding`}>{item.status}</div>
            )
        },
        {
            name: 'Port',
            key: 'port',
            fieldName: 'port',
            minWidth: 60,
            maxWidth: 90,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <div className='succeed-padding'>
                    <div>{item.port}</div>
                </div>
            )
        },
        {
            name: 'Platform',
            key: 'platform',
            fieldName: 'platform',
            minWidth: 110,
            maxWidth: 130,
            isResizable: true,
            data: 'string',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='commonStyle succeed-padding'>{item.platform}</div>
        },
        {
            name: 'Start time',
            key: 'startTime',
            fieldName: 'startTime',
            minWidth: 140,
            maxWidth: 160,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <div className='succeed-padding'>
                    <div>{formatTimestamp(item.startTime)}</div>
                </div>
            )
        },
        {
            name: 'End time',
            key: 'endTime',
            fieldName: 'endTime',
            minWidth: 120,
            maxWidth: 160,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <div className='succeed-padding'>
                    <div>{formatTimestamp(item.endTime)}</div>
                </div>
            )
        }
    ];

    private clickFilter(_e: any): void {
        const { hideFilter } = this.state;
        if (!hideFilter === true) {
            this.setSearchSource();
        }
        this.setState(() => ({ hideFilter: !hideFilter }));
    }

    private setOriginSource(): void {
        let { originExperimentList } = this.state;
        const { sortInfo } = this.state;
        if (originExperimentList !== undefined) {
            originExperimentList = this.commonSelectString(originExperimentList, '');
            const sortedData = getSortedSource(originExperimentList, sortInfo);
            this.setState(() => ({
                source: sortedData
            }));
        }
    }

    private searchNameAndId(_event, newValue): void {
        const { originExperimentList, sortInfo } = this.state;
        if (newValue !== undefined) {
            if (newValue === '') {
                this.setOriginSource();
            } else {
                let result = originExperimentList.filter(
                    item =>
                        item.experimentName.toLowerCase().includes(newValue.toLowerCase()) ||
                        item.id.toLowerCase().includes(newValue.toLowerCase())
                );
                result = this.commonSelectString(result, '');
                const sortedResult = getSortedSource(result, sortInfo);
                this.setState(() => ({
                    source: sortedResult,
                    searchSource: sortedResult
                }));
            }
        }
    }

    /***
     * status, platform
     * param
     * data: searchSource
     * field: no care selected filed
     */
    private commonSelectString = (data: AllExperimentList[], field: string): AllExperimentList[] => {
        const { selectedStatus, selectedPlatform, selectedStartDate, selectedEndDate } = this.state;
        const hasStatus = selectedStatus === '' ? false : true;
        const hasPlatform = selectedPlatform === '' ? false : true;
        const hasStartDate = selectedStartDate === undefined ? false : true;
        const hasEndDate = selectedEndDate === undefined ? false : true;

        if (field === 'status') {
            if (hasPlatform) {
                data = filterByStatusOrPlatform(selectedPlatform, 'platform', data);
            }
        }
        if (field === 'platform') {
            if (hasStatus) {
                data = filterByStatusOrPlatform(selectedStatus, 'status', data);
            }
        }

        if (field === '') {
            if (hasPlatform) {
                data = filterByStatusOrPlatform(selectedPlatform, 'platform', data);
            }
            if (hasStatus) {
                data = filterByStatusOrPlatform(selectedStatus, 'status', data);
            }
        }

        if (hasStartDate) {
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            data = data.filter(temp => compareDate(new Date(temp.startTime), selectedStartDate!));
        }
        if (hasEndDate) {
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            data = data.filter(temp => compareDate(new Date(temp.endTime), selectedEndDate!));
        }

        return data;
    };

    // status platform startTime endTime
    private selectStatus = (_event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            const { searchSource, sortInfo } = this.state;
            let result = filterByStatusOrPlatform(item.key, 'status', searchSource);
            result = this.commonSelectString(result, 'status');
            this.setState({ selectedStatus: item.key, source: getSortedSource(result, sortInfo) });
        }
    };

    private selectPlatform = (_event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            const { searchSource, sortInfo } = this.state;
            let result = filterByStatusOrPlatform(item.key, 'platform', searchSource);
            result = this.commonSelectString(result, 'platform');
            this.setState({ selectedPlatform: item.key, source: getSortedSource(result, sortInfo) });
        }
    };

    private getSelectedData(type: string, date: Date | null | undefined): void {
        if (date !== null && date !== undefined) {
            const {
                selectedStatus,
                selectedPlatform,
                selectedStartDate,
                selectedEndDate,
                searchSource,
                sortInfo
            } = this.state;
            // 只能set item.key
            const hasStatus = selectedStatus === '' ? false : true;
            const hasPlatform = selectedPlatform === '' ? false : true;
            const hasStartDate = selectedStartDate === undefined ? false : true;
            const hasEndDate = selectedEndDate === undefined ? false : true;
            let result;
            if (type === 'start') {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                result = searchSource.filter(item => compareDate(new Date(item.startTime), date));
                if (hasStatus) {
                    result = result.filter(temp => temp.status === selectedStatus);
                }
                if (hasPlatform) {
                    result = result.filter(temp => temp.platform === selectedPlatform);
                }
                if (hasEndDate) {
                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    result = result.filter(temp => compareDate(new Date(temp.endTime), selectedEndDate!));
                }
                this.setState(() => ({
                    source: getSortedSource(result, sortInfo),
                    selectedStartDate: date
                }));
            } else {
                result = searchSource.filter(item => compareDate(new Date(item.endTime), date));

                if (hasStatus) {
                    result = result.filter(temp => temp.status === selectedStatus);
                }
                if (hasPlatform) {
                    result = result.filter(temp => temp.platform === selectedPlatform);
                }
                if (hasStartDate) {
                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    result = result.filter(temp => compareDate(new Date(temp.startTime), selectedStartDate!));
                }
                this.setState(() => ({
                    source: getSortedSource(result, sortInfo),
                    selectedEndDate: date
                }));
            }
        }
    }

    // reset
    private setSearchSource(): void {
        const { searchSource, sortInfo } = this.state;
        this.setState(() => ({
            source: getSortedSource(searchSource, sortInfo),
            selectedStatus: '',
            selectedPlatform: '',
            selectedStartDate: undefined,
            selectedEndDate: undefined
        }));
    }
}

export default Experiment;
