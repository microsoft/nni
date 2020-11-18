import * as React from 'react';
import { Stack, DetailsList, DefaultButton, Icon, SearchBox, Dropdown, DatePicker, DayOfWeek, TooltipHost, DirectionalHint } from '@fluentui/react';
import { formatTimestamp } from '../../static/function';
import { TOOLTIP_BACKGROUND_COLOR, EXPERIMENTSTATUS, PLATFORM } from '../../static/const';
import DayPickerStrings from './experimentConst';
import '../../App.scss';
import '../../static/style/experiment/experiment.scss';
import '../../static/style/common/ellipsis.scss';
import '../../static/style/tableStatus.css';
const data = require('./experiment.json');

interface AllExperimentList {
    name: string;
    id: string;
    status: string;
    port: number;
    platform: string;
    startTime: number;
    endTime: number;
    tag: string[];
}

interface OverviewState {
    hideFilter: boolean;
    searchInputVal: string;
    selectedStatus: string;
    selectedPlatform: string;
    selectedStartDate?: Date;
    selectedEndDate?: Date;
    source: Array<AllExperimentList>;
    filterSource: Array<AllExperimentList>;
    filterSourceOrigin: Array<AllExperimentList>;
}

class Experiment extends React.Component<{}, OverviewState> {
    constructor(props) {
        super(props);
        this.state = {
            hideFilter: true,
            searchInputVal: '',
            selectedStatus: '',
            selectedPlatform: '',
            source: data,
            filterSource: data,
            filterSourceOrigin: data
        };
    }

    render(): React.ReactNode {
        const { hideFilter, selectedStatus, source, selectedPlatform, selectedStartDate, selectedEndDate } = this.state;
        return (
            <Stack className='contentBox expBackground'>
                <Stack className='content'>
                    <Stack className='experimentList'>
                        <Stack className='box' horizontal>
                            <div className='search'>
                                <SearchBox
                                    className='search-input'
                                    placeholder="Search the experiment by name, ID, tags..."
                                    onEscape={(_ev): void => {

                                        this.setState(() => ({ source: data }));
                                        console.log('Custom onEscape Called');
                                    }}
                                    onClear={(_ev): void => {
                                        this.setState(() => ({ source: data }));
                                        console.log('onClear');
                                        // 点 × 操作
                                        console.info('source', this.state.source);
                                    }}
                                    onChange={this.searchNameAndId.bind(this)}
                                />
                            </div>
                            <div className='filter'>
                                <DefaultButton
                                    onClick={this.clickFilter.bind(this)}
                                    // className='fiter-button'
                                    className={`${!hideFilter ? 'filter-button-open' : null}`}
                                >
                                    <Icon iconName='Equalizer' />
                                    <span className='margin'>Filter</span>
                                </DefaultButton>
                            </div>
                        </Stack>
                        <Stack className={`${hideFilter ? 'hidden' : ''} filter-condition`} horizontal gap={25}>
                            <Dropdown
                                label="Status"
                                selectedKey={selectedStatus}
                                onChange={this.selectStatus.bind(this)}
                                placeholder="Select an option"
                                options={this.fillOptions(EXPERIMENTSTATUS)}
                                className='filter-condition-status'
                            />
                            <Dropdown
                                label="Platform"
                                selectedKey={selectedPlatform}
                                onChange={this.selectPlatform.bind(this)}
                                placeholder="Select an option"
                                options={this.fillOptions(PLATFORM)}
                                className='filter-condition-platform'
                            />
                            <DatePicker
                                label='Start time'
                                firstDayOfWeek={DayOfWeek.Sunday}
                                strings={DayPickerStrings}
                                showMonthPickerAsOverlay={true}
                                placeholder="Select a date..."
                                ariaLabel="Select a date"
                                value={selectedStartDate}
                                // dateTimeFormatter={formatMonthDayYear()}
                                // formatDate={(date?: Date): string => date!.toString()}
                                onSelectDate={this.getSelectedData.bind(this, 'start')}
                            />
                            <DatePicker
                                label='End time'
                                firstDayOfWeek={DayOfWeek.Sunday}
                                strings={DayPickerStrings}
                                showMonthPickerAsOverlay={true}
                                placeholder="Select a date..."
                                ariaLabel="Select a date"
                                value={selectedEndDate}
                                onSelectDate={this.getSelectedData.bind(this, 'end')}
                            />
                            <DefaultButton
                                onClick={this.setOriginSource.bind(this)}
                                className='reset'
                            >
                                <Icon iconName='Refresh' />
                                <span className='margin'>Reset</span>
                            </DefaultButton>
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
        );
    }

    private fillOptions(arr: string[]): any {
        const list: Array<object> = [];

        arr.map(item => {
            list.push({ key: item, text: item });
        });

        return list;
    }

    // TODO: 引入tag之后调整大小屏幕下的列宽
    private columns = [
        {
            name: 'Name',
            key: 'name',
            fieldName: 'name', // required!
            minWidth: 50,
            maxWidth: 87,
            isResizable: true,
            data: 'number',
            // onColumnClick: this.onColumnClick,
            // onClick={() => {window.open(webuiPortal)}}
            onRender: (item: any): React.ReactNode => {
                const hostname = window.location.hostname;
                const protocol = window.location.protocol;
                const webuiPortal = `${protocol}//${hostname}:${item.port}/oview`;
                return (
                    <div className='succeed-padding ellipsis'>
                        <a href={webuiPortal} className='link' target='_blank' rel='noopener noreferrer'>
                            {item.name}
                        </a>
                    </div>
                );
            }
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
            // onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding id'>{item.id}</div>
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
            name: 'Port',
            key: 'port',
            fieldName: 'port',
            minWidth: 65,
            maxWidth: 90,
            isResizable: true,
            data: 'number',
            // onColumnClick: this.onColumnClick,
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
            minWidth: 80,
            maxWidth: 100,
            isResizable: true,
            data: 'string',
            // onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => (
                <div className='commonStyle succeed-padding'>{item.platform}</div>
            )
        },
        {
            name: 'Start time',
            key: 'startTime',
            fieldName: 'startTime',
            minWidth: 100,
            maxWidth: 160,
            isResizable: true,
            data: 'number',
            // onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>
                <div>{formatTimestamp(item.startTime)}</div>
            </div>
        },
        {
            name: 'End time',
            key: 'endTime',
            fieldName: 'endTime',
            minWidth: 100,
            maxWidth: 160,
            isResizable: true,
            data: 'number',
            // onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>
                <div>{formatTimestamp(item.endTime)}</div>
            </div>
        },
        {
            name: 'Tag',
            key: 'tag',
            fieldName: 'tag',
            minWidth: 100,
            maxWidth: 160,
            isResizable: true,
            data: 'number',
            // onColumnClick: this.onColumnClick,
            onRender: (item: any): React.ReactNode => {
                return (
                    <TooltipHost
                        content={item.tag.join(', ')}
                        className='ellipsis'
                        directionalHint={DirectionalHint.bottomCenter}
                        tooltipProps={{
                            calloutProps: {
                                styles: {
                                    beak: { background: TOOLTIP_BACKGROUND_COLOR },
                                    beakCurtain: { background: TOOLTIP_BACKGROUND_COLOR },
                                    calloutMain: { background: TOOLTIP_BACKGROUND_COLOR }
                                }
                            }
                        }}
                    >
                        <div className='succeed-padding tagContainer'>
                            {
                                item.tag.map(tag => {
                                    return (
                                        <span className='tag' key={tag}>{tag}</span>
                                    );
                                })
                            }
                        </div>
                    </TooltipHost>
                );
            }
        }
    ];

    private clickFilter(_e: any): void {
        const { hideFilter } = this.state;
        this.setOriginSource();
        this.setState(() => ({ hideFilter: !hideFilter }));

    }

    private searchNameAndId(_event, newValue): void {
        if (newValue !== undefined) {
            // 空格回退操作
            if (newValue === '') {
                this.setState(() => ({ source: data, filterSource: data }));
                console.info('source', this.state.source);
                return;
            }
            const result = this.state.source.filter(item => (item.name.includes(newValue) ||
                item.id.includes(newValue) || item.tag.join(',').includes(newValue)));
            this.setState(() => ({
                source: result, filterSource: result, filterSourceOrigin: result
            }));
        }
    }

    // status platform startTime endTime
    selectStatus = (_event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            const { selectedPlatform, selectedStartDate, selectedEndDate, filterSource } = this.state;
            // 只能set item.key
            const hasPlatform = selectedPlatform === '' ? false : true;
            const hasStartDate = selectedStartDate === undefined ? false : true;
            const hasEndDate = selectedEndDate === undefined ? false : true;
            let result;
            result = filterSource.filter(temp => (temp.status === item.key));
            if (hasPlatform) {
                result = result.filter(temp => (temp.platform === selectedPlatform));
            }
            if (hasStartDate) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                result = result.filter(temp => this.compareDate(new Date(temp.startTime), selectedStartDate!));
            }
            if (hasEndDate) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                result = result.filter(temp => this.compareDate(new Date(temp.endTime), selectedEndDate!));
            }
            this.setState({ selectedStatus: item.key, source: result });
        }
    };

    selectPlatform = (_event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            const { selectedStatus, selectedStartDate, selectedEndDate, filterSource } = this.state;
            // 只能set item.key
            const hasStatus = selectedStatus === '' ? false : true;
            const hasStartDate = selectedStartDate === undefined ? false : true;
            const hasEndDate = selectedEndDate === undefined ? false : true;
            let result;
            result = filterSource.filter(temp => (temp.platform === item.key));
            if (hasStatus) {
                result = result.filter(temp => (temp.status === selectedStatus));
            }
            if (hasStartDate) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                result = result.filter(temp => this.compareDate(new Date(temp.startTime), selectedStartDate!));
            }
            if (hasEndDate) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                result = result.filter(temp => this.compareDate(new Date(temp.endTime), selectedEndDate!));
            }
            this.setState({ selectedPlatform: item.key, source: result });
        }
    };

    private compareDate(date1: Date, date2: Date): boolean {
        if (date1.getFullYear() === date2.getFullYear()) {
            if (date1.getMonth() === date2.getMonth()) {
                if (date1.getDate() === date2.getDate()) {
                    return true;
                }
            }
        }

        return false;
    }

    private getSelectedData(type: string, date: Date | null | undefined): void {
        if (date !== null && date !== undefined) {
            const { selectedStatus, selectedPlatform, selectedStartDate, selectedEndDate, filterSource } = this.state;
            // 只能set item.key
            const hasStatus = selectedStatus === '' ? false : true;
            const hasPlatform = selectedStatus === '' ? false : true;
            const hasStartDate = selectedStartDate === undefined ? false : true;
            const hasEndDate = selectedEndDate === undefined ? false : true;
            let result;

            if (type === 'start') {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                result = filterSource.filter(item => this.compareDate(new Date(item.startTime), date));

                if (hasStatus) {
                    result = result.filter(temp => (temp.status === selectedStatus));
                }

                if (hasPlatform) {
                    result = result.filter(temp => (temp.platform === selectedPlatform));
                }

                if (hasEndDate) {
                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    result = result.filter(temp => this.compareDate(new Date(temp.endTime), selectedEndDate!));
                }

                this.setState({
                    source: result,
                    selectedStartDate: date
                });
            } else {
                result = filterSource.filter(item => this.compareDate(new Date(item.endTime), date));

                if (hasStatus) {
                    result = result.filter(temp => (temp.status === selectedStatus));
                }

                if (hasPlatform) {
                    result = result.filter(temp => (temp.platform === selectedPlatform));
                }

                if (hasStartDate) {
                    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                    result = result.filter(temp => this.compareDate(new Date(temp.startTime), selectedEndDate!));
                }

                this.setState({
                    source: result,
                    selectedEndDate: date
                });
            }

        }
    }

    // reset
    private setOriginSource(): void {
        this.setState(() => ({
            source: this.state.filterSource,
            selectedStatus: '',
            selectedPlatform: '',
            selectedStartDate: undefined,
            selectedEndDate: undefined
        }));
    }
}

export default Experiment;
