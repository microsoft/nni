import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Stack, PrimaryButton } from '@fluentui/react';
import { searchConditonsGap } from '../../modals/ChildrenGap';

// This file is for search trial ['Trial id', 'Trial No.']

function GeneralSearch(props): any {
    const { idOrTrialNo, changeSearchFilterList, setSearchInputVal, updatePage } = props;
    const [firstInputVal, setFirstInputVal] = useState('');

    function _updateFirstInputVal(ev: React.ChangeEvent<HTMLInputElement>): void {
        setFirstInputVal(ev.target.value);
    }

    function apply(): void {
        const { searchFilter } = props;
        const temp = JSON.parse(JSON.stringify(searchFilter));
        const find = temp.filter(item => item.name === idOrTrialNo);
        if (find.length > 0) {
            temp.forEach(item => {
                if (item.name === idOrTrialNo) {
                    item.value1 = firstInputVal;
                }
            });
        } else {
            temp.push({
                name: idOrTrialNo,
                value1: firstInputVal
            });
        }
        setSearchInputVal(getFiterConditionString(temp));
        changeSearchFilterList(temp);
        updatePage();
    }

    function getFiterConditionString(searchFilter): string {
        let str = '';
        searchFilter.forEach(item => {
            if (item.name === 'Trial id') {
                str = str + `Trial id:${item.value1}; `;
            } else {
                str = str + `Trial No.:${item.value1}; `;
            }
        });
        return str;
    }

    return (
        // id & No
        <Stack horizontal className='filterConditions' tokens={searchConditonsGap}>
            <span>{idOrTrialNo === 'Trial id' ? 'Includes' : 'Equals to'}</span>
            <input type='text' className='input input-padding' placeholder='xxx' onChange={_updateFirstInputVal} />
            <PrimaryButton text='Apply' className='btn-vertical-middle' onClick={apply} />
        </Stack>
    );
}

GeneralSearch.propTypes = {
    idOrTrialNo: PropTypes.string,
    searchFilter: PropTypes.array,
    setSearchInputVal: PropTypes.func,
    changeSearchFilterList: PropTypes.func,
    updatePage: PropTypes.func
};

export default GeneralSearch;
