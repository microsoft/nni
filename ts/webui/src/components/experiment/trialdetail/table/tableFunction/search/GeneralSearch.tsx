import React, { useState, useContext } from 'react';
import PropTypes from 'prop-types';
import { Stack, PrimaryButton } from '@fluentui/react';
import { gap10 } from '@components/fluent/ChildrenGap';
import { AppContext } from '@/App';
import { getSearchInputValueBySearchList } from './searchFunction';

// This file is for search trial ['Trial id', 'Trial No.']
// you could use `filter` button -> `Trial id` or `Trial No.` to input value
// if you want to input something in the search input, please follow this fomat
// Trial id:r; // filter all trials that trial id include `r`
// Trial No.:1;  // only filter one trial that trial no is 1.
// And if you want to filter trials id|No by using simple function, you could use this:(nni version >= 2.8 support simple search)
// 1 // only filter one trial that trial no is 1
// bw // only filter that trial id include `bw`

function GeneralSearch(props): any {
    const { updateDetailPage } = useContext(AppContext);
    // searchName val: Trial No. | Trial id
    const { searchName, searchFilter, dismiss, changeSearchFilterList, setSearchInputVal } = props;
    const [firstInputVal, setFirstInputVal] = useState(getSearchNameInit());

    function updateFirstInputVal(ev: React.ChangeEvent<HTMLInputElement>): void {
        setFirstInputVal(ev.target.value);
    }

    function getSearchNameInit(): string {
        let str = ''; // init ''
        const find = searchFilter.find(item => item.name === searchName);

        if (find !== undefined) {
            str = find.value1; // init by filter value
        }

        return str;
    }

    function startFilterTrial(): void {
        const { searchFilter } = props;
        const searchFilterConditions = JSON.parse(JSON.stringify(searchFilter));
        const find = searchFilterConditions.filter(item => item.name === searchName);

        if (firstInputVal === '') {
            alert('Please input related value!');
            return;
        }

        if (find.length > 0) {
            // change this record
            // Trial id | Trial No. only need {search name, search value} these message
            searchFilterConditions.forEach(item => {
                if (item.name === searchName) {
                    item.value1 = firstInputVal;
                    // item.operator = '';
                    item.isChoice = false;
                }
            });
        } else {
            searchFilterConditions.push({
                name: searchName,
                // operator: '',
                value1: firstInputVal,
                isChoice: false
            });
        }
        setSearchInputVal(getSearchInputValueBySearchList(searchFilterConditions));
        changeSearchFilterList(searchFilterConditions);
        updateDetailPage();
        dismiss(); // close menu
    }

    return (
        // Trial id & Trial No.
        <Stack horizontal className='filterConditions' tokens={gap10}>
            <span>{searchName === 'Trial id' ? 'Includes' : 'Equals to'}</span>
            <input type='text' className='input input-padding' onChange={updateFirstInputVal} value={firstInputVal} />
            <PrimaryButton text='Apply' className='btn-vertical-middle' onClick={startFilterTrial} />
        </Stack>
    );
}

GeneralSearch.propTypes = {
    searchName: PropTypes.string,
    searchFilter: PropTypes.array,
    dismiss: PropTypes.func,
    setSearchInputVal: PropTypes.func,
    changeSearchFilterList: PropTypes.func
};

export default GeneralSearch;
