import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Stack, PrimaryButton, Dropdown, IDropdownOption } from '@fluentui/react';
import { trialJobStatus } from '../../../static/const';
import { getFiterConditionString } from '../../../static/function';
import { searchConditonsGap } from '../../modals/ChildrenGap';

// This file is for filtering trial parameters and trial status

function SearchParameterConditions(props): any {
    const { parameter, searchFilter, dismiss, changeSearchFilterList, updatePage, setSearchInputVal } = props;
    const isStatus = parameter === 'StatusNNI';
    const [trialParameterOperatorVal, setTrialParameterOperatorVal] = useState(getInputsVal()[0]);
    const [trialStatusOperator, setTrialStatusOperator] = useState(getInputsVal()[1]);
    const [firstInputVal, setFirstInputVal] = useState(getInputsVal()[1]);
    const [secondInputVal, setSecondInputVal] = useState(getInputsVal()[2]);
    const operatorList = isStatus ? ['=', '≠'] : ['between', '>', '<', '=', '≠'];

    function getInputsVal(): string[] {
        const str: string[] = [];

        if (searchFilter.length > 0) {
            const filterElement = searchFilter.find(ele => ele.name === parameter);
            if (filterElement !== undefined) {
                // set before value [operator, value1, value2]
                str.push(filterElement.operator, filterElement.value1, filterElement.value2);
            } else {
                // set init value
                str.push(`${isStatus ? '=' : 'between'}`, '', '');
            }
        } else {
            str.push(`${isStatus ? '=' : 'between'}`, '', '');
        }

        return str;
    }

    function updateTrialParameterDropdown(
        _event: React.FormEvent<HTMLDivElement>,
        item: IDropdownOption | undefined
    ): void {
        if (item !== undefined) {
            const value = item.key.toString();
            setTrialParameterOperatorVal(value);
        }
    }

    function updateTrialStatusDropdown(
        _event: React.FormEvent<HTMLDivElement>,
        item: IDropdownOption | undefined
    ): void {
        if (item !== undefined) {
            const value = item.key.toString();
            setTrialStatusOperator(value);
            // Status also store in first Input val
            setFirstInputVal(value);
        }
    }

    function updateFirstInputVal(ev: React.ChangeEvent<HTMLInputElement>): void {
        setFirstInputVal(ev.target.value);
    }

    function updateSecondInputVal(ev: React.ChangeEvent<HTMLInputElement>): void {
        setSecondInputVal(ev.target.value);
    }

    // click Apply button
    function startFilterTrials(): void {
        const { searchFilter } = props;
        const newSearchFilters = JSON.parse(JSON.stringify(searchFilter));
        const find = newSearchFilters.filter(ele => ele.name === parameter);
        if (find.length > 0) {
            newSearchFilters.forEach(item => {
                if (item.name === parameter) {
                    item.operator = trialParameterOperatorVal;
                    item.value1 = firstInputVal;
                    item.value2 = secondInputVal;
                }
            });
        } else {
            newSearchFilters.push({
                name: parameter,
                operator: trialParameterOperatorVal,
                value1: firstInputVal,
                value2: secondInputVal
            });
        }
        setSearchInputVal(getFiterConditionString(newSearchFilters));
        changeSearchFilterList(newSearchFilters);
        updatePage();
        dismiss(); // close menu
    }

    return (
        // for trial parameters & Status
        <Stack horizontal className='filterConditions' tokens={searchConditonsGap}>
            <Dropdown
                selectedKey={trialParameterOperatorVal}
                options={operatorList.map(item => ({
                    key: item,
                    text: item
                }))}
                onChange={updateTrialParameterDropdown}
                className='btn-vertical-middle'
                styles={{ root: { width: 100 } }}
            />
            {isStatus ? (
                <Dropdown
                    selectedKey={trialStatusOperator}
                    options={trialJobStatus.map(item => ({
                        key: item,
                        text: item
                    }))}
                    onChange={updateTrialStatusDropdown}
                    className='btn-vertical-middle'
                    styles={{ root: { width: 160 } }}
                />
            ) : (
                <React.Fragment>
                    {trialParameterOperatorVal === 'between' ? (
                        <div>
                            <input
                                type='text'
                                className='input input-padding'
                                // placeholder='Please input value...'
                                onChange={updateFirstInputVal}
                                value={firstInputVal}
                            />
                            <span className='and'>and</span>
                            <input
                                type='text'
                                className='input input-padding'
                                // placeholder='Please input value...'
                                onChange={updateSecondInputVal}
                                value={secondInputVal}
                            />
                        </div>
                    ) : (
                        <input
                            type='text'
                            className='input input-padding'
                            // placeholder='Please input value...'
                            onChange={updateFirstInputVal}
                            value={firstInputVal}
                        />
                    )}
                </React.Fragment>
            )}
            <PrimaryButton text='Apply' className='btn-vertical-middle' onClick={startFilterTrials} />
        </Stack>
    );
}

SearchParameterConditions.propTypes = {
    parameter: PropTypes.string,
    searchFilter: PropTypes.array,
    dismiss: PropTypes.func,
    setSearchInputVal: PropTypes.func,
    changeSearchFilterList: PropTypes.func,
    updatePage: PropTypes.func
};

export default SearchParameterConditions;
