import * as React from 'react';
import { DefaultButton, Icon, Dropdown, DatePicker, DayOfWeek } from '@fluentui/react';
import { EXPERIMENTSTATUS } from '../../static/const';
import { fillOptions } from './expFunction';

interface FilterBtnsProps {
    platform: string[];
    selectedStatus: string;
    selectedPlatform: string;
    selectedStartDate: Date;
    selectedEndDate: Date;
    selectStatus: (_event: React.FormEvent<HTMLDivElement>, item: any) => void;
    selectPlatform: (_event: React.FormEvent<HTMLDivElement>, item: any) => void;
    getSelectedData: (type: string, date: Date | null | undefined) => void;
    setSearchSource: () => void;
}

class FilterBtns extends React.Component<FilterBtnsProps, {}> {
    constructor(props: FilterBtnsProps) {
        super(props);
    }

    render(): React.ReactNode {
        const {
            platform,
            selectedStatus,
            selectedPlatform,
            selectedStartDate,
            selectedEndDate,
            selectStatus,
            selectPlatform,
            getSelectedData,
            setSearchSource
        } = this.props;

        return (
            <React.Fragment>
                <Dropdown
                    label='Status'
                    selectedKey={selectedStatus}
                    onChange={selectStatus.bind(this)}
                    placeholder='Select an option'
                    options={fillOptions(EXPERIMENTSTATUS)}
                    className='filter-condition-status'
                />
                <Dropdown
                    label='Platform'
                    selectedKey={selectedPlatform}
                    onChange={selectPlatform.bind(this)}
                    placeholder='Select an option'
                    options={fillOptions(platform)}
                    className='filter-condition-platform'
                />
                <DatePicker
                    label='Start time'
                    firstDayOfWeek={DayOfWeek.Sunday}
                    showMonthPickerAsOverlay={true}
                    placeholder='Select a date...'
                    ariaLabel='Select a date'
                    value={selectedStartDate}
                    onSelectDate={getSelectedData.bind(this, 'start')}
                />
                <DatePicker
                    label='End time'
                    firstDayOfWeek={DayOfWeek.Sunday}
                    showMonthPickerAsOverlay={true}
                    placeholder='Select a date...'
                    ariaLabel='Select a date'
                    value={selectedEndDate}
                    onSelectDate={getSelectedData.bind(this, 'end')}
                />
                <DefaultButton onClick={setSearchSource.bind(this)} className='reset'>
                    <Icon iconName='Refresh' />
                    <span className='margin'>Reset</span>
                </DefaultButton>
            </React.Fragment>
        );
    }
}

export default FilterBtns;
