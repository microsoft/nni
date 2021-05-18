import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Stack, PrimaryButton } from '@fluentui/react';
import { searchConditonsGap } from '../../modals/ChildrenGap';
import { getFiterConditionString } from '../../../static/function';

// This file is for search trial ['Trial id', 'Trial No.']

function GeneralSearch(props): any {
    const { idOrTrialNo, searchFilter, dismiss, changeSearchFilterList, setSearchInputVal, updatePage } = props;

    const [firstInputVal, setFirstInputVal] = useState(getIdorNoInit());

    function updateFirstInputVal(ev: React.ChangeEvent<HTMLInputElement>): void {
        setFirstInputVal(ev.target.value);
    }

    function getIdorNoInit(): string {
        let str = ''; // init ''

        searchFilter.forEach(item => {
            if (item.name === idOrTrialNo) {
                str = item.value1; // init by filter value
            }
        });

        return str;
    }

    function startFilterTrial(): void {
        const { searchFilter } = props;
        const searchFilterConditions = JSON.parse(JSON.stringify(searchFilter));
        const find = searchFilterConditions.filter(item => item.name === idOrTrialNo);
        if (find.length > 0) {
            // change this record
            searchFilterConditions.forEach(item => {
                if (item.name === idOrTrialNo) {
                    item.value1 = firstInputVal;
                    item.operator = '';
                }
            });
        } else {
            searchFilterConditions.push({
                name: idOrTrialNo,
                operator: '',
                value1: firstInputVal
            });
        }
        setSearchInputVal(getFiterConditionString(searchFilterConditions));
        changeSearchFilterList(searchFilterConditions);
        updatePage();
        dismiss(); // close menu
    }

    return (
        // Trial id & Trial No.
        <Stack horizontal className='filterConditions' tokens={searchConditonsGap}>
            <span>{idOrTrialNo === 'Trial id' ? 'Includes' : 'Equals to'}</span>
            <input
                type='text'
                className='input input-padding'
                // placeholder='Please input value...'
                onChange={updateFirstInputVal}
                value={firstInputVal}
            />
            <PrimaryButton text='Apply' className='btn-vertical-middle' onClick={startFilterTrial} />
        </Stack>
    );
}

GeneralSearch.propTypes = {
    idOrTrialNo: PropTypes.string,
    searchFilter: PropTypes.array,
    dismiss: PropTypes.func,
    setSearchInputVal: PropTypes.func,
    changeSearchFilterList: PropTypes.func,
    updatePage: PropTypes.func
};

export default GeneralSearch;
