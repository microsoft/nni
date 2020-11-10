import * as React from 'react';
import { Stack, DetailsList, DefaultButton, Icon, SearchBox, Dropdown, DatePicker, DayOfWeek } from '@fluentui/react';
import { formatTimestamp } from '../../static/function';
import DayPickerStrings from './experimentConst';
import '../../App.scss';
import '../../static/style/experiment/experiment.scss';
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
    tag: string;
}

interface OverviewState {
    hideFilter: boolean;
    searchInputVal: string;
    selectedStatus: string;
    selectedPlatform: string;
    // selectedStartDate: Date;
    source: Array<AllExperimentList>;
}

class Experiment extends React.Component<{}, OverviewState> {
    constructor(props) {
        super(props);
        this.state = {
            hideFilter: false,
            searchInputVal: '',
            selectedStatus: '',
            selectedPlatform: '',
            // selectedStartDate: '',
            source: data
        };
    }

    render(): React.ReactNode {
        const { hideFilter, selectedStatus, source, selectedPlatform } = this.state;
        return (
            <Stack className='contentBox'>
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
                                    onChange={(_, newValue): void => {
                                        if (newValue !== undefined) {
                                            // 空格回退操作
                                            if (newValue === '') {
                                                this.setState(() => ({ source: data }));
                                                console.info('source', this.state.source);
                                                return;
                                            }
                                            this.setState(() => ({
                                                source: this.state.source.filter(item => (item.status.includes(newValue) ||
                                                    item.id.includes(newValue)))
                                            }));
                                        }
                                        console.log('SearchBox onChange fired: ' + newValue)
                                    }
                                    }
                                // onSearch={(newValue): void => console.log('SearchBox onSearch fired: ' + newValue)}
                                />
                            </div>
                            <div className='filter'>
                                <DefaultButton
                                    onClick={this.clickFilter.bind(this)}
                                    className='fiter-button'
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
                                options={this.statusOption}
                            />
                            <Dropdown
                                label="Platform"
                                selectedKey={selectedPlatform}
                                onChange={this.selectPlatform.bind(this)}
                                placeholder="Select an option"
                                options={this.platformOption}
                            // styles={dropdownStyles}
                            />
                            <DatePicker
                                label='Start time'
                                // className={controlClass.control}
                                firstDayOfWeek={DayOfWeek.Sunday}
                                strings={DayPickerStrings}
                                showMonthPickerAsOverlay={true}
                                placeholder="Select a date..."
                                ariaLabel="Select a date"
                                // dateTimeFormatter={formatMonthDayYear()}
                                // formatDate={(date?: Date): string => date!.toString()}
                                onSelectDate={this.getSelectedData.bind(this, 'start')}
                            />
                            <DatePicker
                                label='End time'
                                // className={controlClass.control}
                                firstDayOfWeek={DayOfWeek.Sunday}
                                strings={DayPickerStrings}
                                placeholder="Select a date..."
                                ariaLabel="Select a date"
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
                            className='succTable'
                        />
                    </Stack>
                </Stack>
            </Stack>
        );
    }

    private statusOption = [
        { key: 'FAILED', text: 'FAILED' },
        { key: 'SUCCEEDED', text: 'SUCCEEDED' },
        { key: 'RUNNING', text: 'RUNNING' }

    ];
    private platformOption = [
        { key: 'local', text: 'local' },
        { key: 'pai', text: 'pai' },
        { key: 'remote', text: 'remote' }

    ];
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
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.name}</div>
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
            onRender: (item: any): React.ReactNode => <div className='succeed-padding'>{item.id}</div>
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
                    <div className='succeed-padding'>
                        {
                            item.tag.map(tag => {
                                return (
                                    <span className='tag' key={tag}>{tag}</span>
                                );
                            })
                        }
                    </div>
                );
            }
        }
    ];

    private clickFilter(_e: any): void {
        const { hideFilter } = this.state;
        this.setState(() => ({ hideFilter: !hideFilter }));
    }

    selectStatus = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            // 只能set item.key
            this.setState({ selectedStatus: item.key, source: this.state.source.filter(temp => temp.status === item.key) });
        }
    };

    selectPlatform = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            // 只能set item.key
            this.setState({ selectedPlatform: item.key, source: this.state.source.filter(temp => temp.platform === item.key) });
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
            const { source } = this.state;
            if (type === 'start') {
                this.setState({ source: source.filter(item => this.compareDate(new Date(item.startTime), date)) });
            } else {
                this.setState({ source: source.filter(item => this.compareDate(new Date(item.endTime), date)) });
            }
        }

    }
    private setOriginSource(): void {
        this.setState(() => ({ source: data, selectedStatus: '', selectedPlatform: '' }));
    }
}

export default Experiment;
