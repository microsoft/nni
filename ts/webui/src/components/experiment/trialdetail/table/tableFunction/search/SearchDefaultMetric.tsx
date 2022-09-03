import React, { useState, useContext } from 'react';
import { Stack, PrimaryButton, Dropdown, IDropdownOption } from '@fluentui/react';
import { EXPERIMENT } from '@static/datamodel';
import { getSearchInputValueBySearchList } from './searchFunction';
import { gap10 } from '@components/fluent/ChildrenGap';
import { AppContext } from '@/App';
import { SearchItems } from '@static/interface';

// This file is for filtering trial default metric column including intermediate results
// you could click `filter` button -> `Default metric` use it
// if you want to use search input , please use these format:
// Default metric>0.9; // search trial metric value > 0.9
// Default metric<0.9; // search trial metric value < 0.9
// Default metric:[0.1,0.2];  //  0.1 < trial metric < 0.2
// Default metric:0.1009; // trial metric = 0.1009, because shown metric is dealed with,so it's no use in most time

interface SearchDefaultMetricProps {
    parameter: string;
    searchFilter: SearchItems[];
    dismiss: () => void;
    setSearchInputVal: (a: string) => void;
    changeSearchFilterList: (a: SearchItems[]) => void;
    updatePage: () => void;
}

function SearchDefaultMetric(props: SearchDefaultMetricProps): any {
    const { parameter, searchFilter, dismiss, changeSearchFilterList, setSearchInputVal } = props;
    const { updateDetailPage } = useContext(AppContext);
    const operatorList = ['between', '>', '<', '='];

    const initValueList = getInitVal();
    const [operatorVal, setOperatorVal] = useState(initValueList[0] as string);
    const [firstInputVal, setFirstInputVal] = useState(initValueList[1] as string);
    const [secondInputVal, setSecondInputVal] = useState(initValueList[2] as string);

    function getInitVal(): Array<string | string[]> {
        // push value: operator, firstInputVal(value1), secondInputVal(value2), choiceValue
        const str: Array<string | string[]> = [];

        if (searchFilter.length > 0) {
            const filterElement = searchFilter.find(ele => ele.name === parameter);
            if (filterElement !== undefined) {
                str.push(filterElement.operator, filterElement.value1.toString(), filterElement.value2.toString());
            } else {
                // set init value
                str.push('between', '', '', [] as string[]);
            }
        } else {
            str.push('between', '', '', [] as string[]);
        }

        return str;
    }

    function updateOperatorDropdown(_event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void {
        if (item !== undefined) {
            setOperatorVal(item.key.toString());
        }
    }

    function updateFirstInputVal(ev: React.ChangeEvent<HTMLInputElement>): void {
        setFirstInputVal(ev.target.value);
    }

    function updateSecondInputVal(ev: React.ChangeEvent<HTMLInputElement>): void {
        setSecondInputVal(ev.target.value);
    }

    function getSecondInputVal(): string {
        if (secondInputVal === '' && operatorVal === 'between') {
            // if user uses 'between' operator and doesn't write the second input value,
            // help to set second value as this parameter max value
            return String(EXPERIMENT.searchSpace[parameter]._value[1]);
        }

        return secondInputVal as string;
    }

    // click Apply button
    function startFilterTrials(): void {
        if (firstInputVal === '') {
            alert('Please input related value!');
            return;
        }

        if (firstInputVal.match(/[a-zA-Z]/) || secondInputVal.match(/[a-zA-Z]/)) {
            alert('Please input a number!');
            return;
        }

        let newSearchFilters = JSON.parse(JSON.stringify(searchFilter));
        const find = newSearchFilters.filter(ele => ele.name === parameter);

        if (find.length > 0) {
            newSearchFilters = newSearchFilters.filter(item => item.name !== parameter);
        } else {
            newSearchFilters.push({
                name: parameter,
                operator: operatorVal,
                value1: firstInputVal,
                value2: getSecondInputVal(),
                isChoice: false
            });
        }

        setSearchInputVal(getSearchInputValueBySearchList(newSearchFilters));
        changeSearchFilterList(newSearchFilters);
        updateDetailPage();
        dismiss(); // close menu
    }

    return (
        <Stack horizontal className='filterConditions' tokens={gap10}>
            <Dropdown
                selectedKey={operatorVal}
                options={operatorList.map(item => ({
                    key: item,
                    text: item
                }))}
                onChange={updateOperatorDropdown}
                className='btn-vertical-middle'
                styles={{ root: { width: 100 } }}
            />
            {operatorVal === 'between' ? (
                <div>
                    <input
                        type='text'
                        className='input input-padding'
                        onChange={updateFirstInputVal}
                        value={firstInputVal}
                    />
                    <span className='and'>and</span>
                    <input
                        type='text'
                        className='input input-padding'
                        onChange={updateSecondInputVal}
                        value={secondInputVal}
                    />
                </div>
            ) : (
                <input
                    type='text'
                    className='input input-padding'
                    onChange={updateFirstInputVal}
                    value={firstInputVal}
                />
            )}
            <PrimaryButton text='Apply' className='btn-vertical-middle' onClick={startFilterTrials} />
        </Stack>
    );
}

export default SearchDefaultMetric;
