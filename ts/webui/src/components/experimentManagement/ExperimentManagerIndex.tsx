import * as React from 'react';
import {
    Stack,
    DetailsList,
    DefaultButton,
    Icon,
    SearchBox,
    IColumn,
    IStackTokens,
    MessageBar,
    MessageBarType
} from '@fluentui/react';
import { ExperimentsManager } from '@model/experimentsManager';
import { expformatTimestamp, copyAndSort } from '@static/function';
import { AllExperimentList, SortInfo } from '@static/interface';
import { compareDate, filterByStatusOrPlatform, getSortedSource } from './expFunction';
import { MAXSCREENCOLUMNWIDHT, MINSCREENCOLUMNWIDHT } from './experimentConst';
import { Hearder } from './Header';
import TrialIdColumn from './TrialIdColumn';
import FilterBtns from './FilterBtns';
import { TitleContext } from '../title/TitleContext';
import { Title } from '../title/Title';
import '@style/App.scss';
import '@style/nav/nav.scss';
import '@style/common/common.scss';
import '@style/common/experimentStatusColor.scss';
import '@style/common/trialStatus.css';
import '@style/experimentManagement/experiment.scss';

const expTokens: IStackTokens = {
    childrenGap: 25
};

interface ExpListState {
    columns: IColumn[];
    platform: string[];
    errorMessage: string;
    hideFilter: boolean;
    searchInputVal: string;
    selectedStatus: string[];
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
            platform: [],
            columns: this.columns,
            errorMessage: '',
            hideFilter: true,
            searchInputVal: '',
            selectedStatus: [],
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
            searchSource: result,
            platform: EXPERIMENTMANAGER.getPlatformList(),
            errorMessage: EXPERIMENTMANAGER.getExpErrorMessage()
        }));
    }

    render(): React.ReactNode {
        const {
            platform,
            hideFilter,
            selectedStatus,
            source,
            selectedPlatform,
            selectedStartDate,
            selectedEndDate,
            errorMessage
        } = this.state;
        return (
            <Stack className='nni experiments-info' style={{ minHeight: window.innerHeight }}>
                <Hearder />
                {errorMessage !== '' ? (
                    <div className='warning'>
                        <MessageBar messageBarType={MessageBarType.error} isMultiline={true} style={{ width: 400 }}>
                            {errorMessage}
                        </MessageBar>
                    </div>
                ) : null}
                <Stack className='contentBox expBackground'>
                    {/* 64px: navBarHeight; 48: marginTop & Bottom */}
                    <Stack className='content' styles={{ root: { minHeight: window.innerHeight - 112 } }}>
                        <Stack className='experimentList'>
                            <TitleContext.Provider value={{ text: 'All experiments', icon: 'CustomList' }}>
                                <Title />
                            </TitleContext.Provider>
                            <Stack className='box' horizontal>
                                <div className='search'>
                                    <SearchBox
                                        className='search-input'
                                        placeholder='Search the experiment by name or ID'
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
                            <Stack
                                className={`${hideFilter ? 'hidden' : ''} filter-condition`}
                                horizontal
                                tokens={expTokens}
                            >
                                <FilterBtns
                                    platform={platform}
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
            minWidth: MINSCREENCOLUMNWIDHT,
            maxWidth: MAXSCREENCOLUMNWIDHT,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div>{item.experimentName}</div>
        },
        {
            name: 'ID',
            key: 'id',
            fieldName: 'id',
            minWidth: MINSCREENCOLUMNWIDHT,
            maxWidth: MAXSCREENCOLUMNWIDHT,
            isResizable: true,
            className: 'tableHead leftTitle',
            data: 'string',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <TrialIdColumn item={item} />
        },
        {
            name: 'Status',
            key: 'status',
            fieldName: 'status',
            minWidth: MINSCREENCOLUMNWIDHT,
            maxWidth: MAXSCREENCOLUMNWIDHT,
            isResizable: true,
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className={`${item.status} commonStyle`}>{item.status}</div>
        },
        {
            name: 'Port',
            key: 'port',
            fieldName: 'port',
            minWidth: MINSCREENCOLUMNWIDHT - 15,
            maxWidth: MAXSCREENCOLUMNWIDHT - 30,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <div className={item.status === 'STOPPED' ? 'gray-port' : ''}>
                    {item.port !== undefined ? item.port : '--'}
                </div>
            )
        },
        {
            name: 'Platform',
            key: 'platform',
            fieldName: 'platform',
            minWidth: MINSCREENCOLUMNWIDHT - 15,
            maxWidth: MAXSCREENCOLUMNWIDHT - 30,
            isResizable: true,
            data: 'string',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='commonStyle'>{item.platform}</div>
        },
        {
            name: 'Start time',
            key: 'startTime',
            fieldName: 'startTime',
            minWidth: MINSCREENCOLUMNWIDHT + 15,
            maxWidth: MAXSCREENCOLUMNWIDHT + 30,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div>{expformatTimestamp(item.startTime)}</div>
        },
        {
            name: 'End time',
            key: 'endTime',
            fieldName: 'endTime',
            minWidth: MINSCREENCOLUMNWIDHT + 15,
            maxWidth: MAXSCREENCOLUMNWIDHT + 30,
            isResizable: true,
            data: 'number',
            onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div>{expformatTimestamp(item.endTime)}</div>
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
                const searchInput = newValue.trim();
                let result = originExperimentList.filter(
                    item =>
                        (item.experimentName !== null &&
                            item.experimentName.toLowerCase().includes(searchInput.toLowerCase())) ||
                        item.id.toLowerCase().includes(searchInput.toLowerCase())
                );
                result = this.commonSelectString(result, '');
                const sortedResult = getSortedSource(result, sortInfo);
                this.setState(() => ({
                    source: sortedResult,
                    searchSource: sortedResult
                }));
            }
            this.setState(() => ({
                searchInputVal: newValue.trim()
            }));
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

        if (field === 'status') {
            data = filterByStatusOrPlatform(selectedPlatform, 'platform', data);
        }
        if (field === 'platform') {
            data = filterByStatusOrPlatform(selectedStatus, 'status', data);
        }

        if (field === '') {
            data = Array.from(
                new Set([
                    ...filterByStatusOrPlatform(selectedPlatform, 'platform', data),
                    ...filterByStatusOrPlatform(selectedStatus, 'status', data)
                ])
            );
        }

        data = data.filter(
            item =>
                (selectedStartDate === undefined || compareDate(new Date(item.startTime), selectedStartDate)) &&
                (selectedEndDate === undefined || compareDate(new Date(item.endTime), selectedEndDate))
        );

        return data;
    };

    // status platform startTime endTime
    private selectStatus = (_event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            const { searchSource, sortInfo, selectedStatus } = this.state;
            const newSelectedStatus = item.selected
                ? [...selectedStatus, item.key as string]
                : selectedStatus.filter(key => key !== item.key);
            let result = filterByStatusOrPlatform(newSelectedStatus, 'status', searchSource);
            result = this.commonSelectString(result, 'status');
            this.setState({
                selectedStatus: newSelectedStatus,
                source: getSortedSource(result, sortInfo)
            });
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
            const { selectedStatus, selectedPlatform, selectedStartDate, selectedEndDate, searchSource, sortInfo } =
                this.state;
            const hasPlatform = selectedPlatform === '' ? false : true;

            // filter status, platform
            let result = filterByStatusOrPlatform(selectedStatus, 'status', searchSource);
            if (hasPlatform) {
                result = result.filter(temp => temp.platform === selectedPlatform);
            }

            if (type === 'start') {
                result = result.filter(
                    item =>
                        compareDate(new Date(item.startTime), date) &&
                        (selectedEndDate === undefined || compareDate(new Date(item.endTime), selectedEndDate))
                );
                this.setState(() => ({
                    source: getSortedSource(result, sortInfo),
                    selectedStartDate: date
                }));
            } else {
                result = result.filter(
                    item =>
                        compareDate(new Date(item.endTime), date) &&
                        (selectedStartDate === undefined || compareDate(new Date(item.startTime), selectedStartDate))
                );
                this.setState(() => ({
                    source: getSortedSource(result, sortInfo),
                    selectedEndDate: date
                }));
            }
        }
    }

    // reset
    private setSearchSource(): void {
        const { sortInfo, originExperimentList } = this.state;
        let { searchInputVal } = this.state;
        let result = JSON.parse(JSON.stringify(originExperimentList));
        searchInputVal = searchInputVal.trim();
        // user input some value to filter trial [name, id] first...
        if (searchInputVal !== '') {
            // reset experiments list to first filter result
            result = originExperimentList.filter(
                item =>
                    item.id.toLowerCase().includes(searchInputVal.toLowerCase()) ||
                    (item.experimentName !== null &&
                        item.experimentName.toLowerCase().includes(searchInputVal.toLowerCase()))
            );
        }
        this.setState(() => ({
            source: getSortedSource(result, sortInfo),
            selectedStatus: [],
            selectedPlatform: '',
            selectedStartDate: undefined,
            selectedEndDate: undefined
        }));
    }
}

export default Experiment;
