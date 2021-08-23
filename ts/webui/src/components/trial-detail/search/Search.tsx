import React, { useState } from 'react';
import PropTypes from 'prop-types';
import {
    Stack,
    DefaultButton,
    IContextualMenuProps,
    IContextualMenuItem,
    DirectionalHint,
    SearchBox
} from '@fluentui/react';
import { EXPERIMENT } from '../../../static/datamodel';
import { SearchItems } from '../../../static/interface';
import SearchParameterConditions from './SearchParameterConditions';
import GeneralSearch from './GeneralSearch';
import { classNames, isChoiceType } from './searchFunction';

// TableList search layout

function Search(props): any {
    const { searchFilter, changeSearchFilterList, updatePage } = props;
    const [searchInputVal, setSearchInputVal] = useState('');

    function getSearchMenu(parameterList): IContextualMenuProps {
        const menu: Array<object> = [];

        parameterList.unshift('StatusNNI');

        ['Trial id', 'Trial No.'].forEach(item => {
            menu.push({
                key: item,
                text: item,
                subMenuProps: {
                    items: [
                        {
                            key: item,
                            text: item,
                            // component: GeneralSearch.tsx
                            onRender: renderIdAndNoComponent.bind(item)
                        }
                    ]
                }
            });
        });

        parameterList.forEach(item => {
            menu.push({
                key: item,
                text: item === 'StatusNNI' ? 'Status' : item,
                subMenuProps: {
                    items: [
                        {
                            key: item,
                            text: item,
                            // component: SearchParameterConditions.tsx
                            onRender: renderParametersSearchComponent.bind(item)
                        }
                    ]
                }
            });
        });

        const filterMenu: IContextualMenuProps = {
            shouldFocusOnMount: true,
            directionalHint: DirectionalHint.bottomLeftEdge,
            className: classNames.menu,
            items: menu as any
        };

        return filterMenu;
    }

    // Avoid nested experiments, nested experiments do not support hyperparameter search
    const searchMenuProps: IContextualMenuProps = getSearchMenu(
        EXPERIMENT.isNestedExp() ? [] : Object.keys(EXPERIMENT.searchSpace)
    );

    function renderParametersSearchComponent(item: IContextualMenuItem, dismissMenu: () => void): JSX.Element {
        return (
            <SearchParameterConditions
                parameter={item.text}
                searchFilter={searchFilter} // search filter list
                changeSearchFilterList={changeSearchFilterList}
                updatePage={updatePage}
                setSearchInputVal={setSearchInputVal}
                dismiss={dismissMenu} // close menu
            />
        );
    }

    function renderIdAndNoComponent(item: IContextualMenuItem, dismissMenu: () => void): JSX.Element {
        return (
            <GeneralSearch
                searchName={item.text}
                searchFilter={searchFilter} // search fliter list
                changeSearchFilterList={changeSearchFilterList}
                setSearchInputVal={setSearchInputVal}
                updatePage={updatePage}
                dismiss={dismissMenu} // after click Apply button to close menu
            />
        );
    }

    function updateSearchText(_, newValue): void {
        setSearchInputVal(newValue);
    }

    // update TableList page
    function changeTableListPage(searchFilterList: Array<SearchItems>): void {
        changeSearchFilterList(searchFilterList);
        updatePage();
    }

    // deal with the format 1.[x, (space)xx] 2. (space)[x]
    function convertStringArrToList(str: string): string[] {
        const value = str.trim().slice(1, str.trim().length - 1); // delete []
        // delete ""
        const result: string[] = [];

        if (value.includes(',')) {
            const arr = value.split(',');
            arr.forEach(item => {
                if (item !== '') {
                    result.push(item);
                }
            });
            return result;
        } else {
            if (value === '') {
                return result;
            } else {
                return [value];
            }
        }
    }

    // SearchBox onSearch event: Filter based on the filter criteria entered by the user
    function startFilter(): void {
        const regEn = /`~!@#$%^&*()+?"{}.'/im;
        const regCn = /·！#￥（——）：；“”‘、，|《。》？、【】[\]]/im;
        if (regEn.test(searchInputVal) || regCn.test(searchInputVal)) {
            alert('Please delete special characters in the conditions!');
            return;
        }
        // according [input val] to change searchFilter list
        const allFilterConditions = searchInputVal.trim().split(';');
        const newSearchFilter: any = [];

        // delete '' in filter list
        if (allFilterConditions.includes('')) {
            allFilterConditions.splice(
                allFilterConditions.findIndex(item => item === ''),
                1
            );
        }

        allFilterConditions.forEach(eachFilterConditionStr => {
            // input content looks like that: `Trial id:`
            if (
                eachFilterConditionStr.endsWith(':') ||
                eachFilterConditionStr.endsWith('<') ||
                eachFilterConditionStr.endsWith('>') ||
                eachFilterConditionStr.endsWith('≠')
            ) {
                return;
            } else {
                let eachFilterConditionArr: string[] = [];

                // EXPERIMENT.searchSpace[parameter]._type === 'choice'
                if (eachFilterConditionStr.includes('>' || '<')) {
                    const operator = eachFilterConditionStr.includes('>') === true ? '>' : '<';
                    eachFilterConditionArr = eachFilterConditionStr.trim().split(operator);
                    newSearchFilter.push({
                        name: eachFilterConditionArr[0],
                        operator: operator,
                        value1: eachFilterConditionArr[1].trim(),
                        value2: '',
                        choice: [],
                        isChoice: false
                    });
                } else if (eachFilterConditionStr.includes('≠')) {
                    // drop_rate≠6; status≠[x,xx,xxx]; conv_size≠[3,7]
                    eachFilterConditionArr = eachFilterConditionStr.trim().split('≠');
                    const filterName = eachFilterConditionArr[0] === 'Status' ? 'StatusNNI' : eachFilterConditionArr[0];
                    const isChoicesType = isChoiceType(filterName);
                    newSearchFilter.push({
                        name: filterName,
                        operator: '≠',
                        value1: isChoicesType ? '' : JSON.parse(eachFilterConditionArr[1].trim()),
                        value2: '',
                        choice: isChoicesType ? convertStringArrToList(eachFilterConditionArr[1]) : [],
                        isChoice: isChoicesType ? true : false
                    });
                } else if (eachFilterConditionStr.includes(':')) {
                    // = : conv_size:[1,2,3,4]; Trial id:3; hidden_size:[1,2], status:[val1,val2,val3]
                    eachFilterConditionArr = eachFilterConditionStr.trim().split(':');
                    const filterName = eachFilterConditionArr[0] === 'Status' ? 'StatusNNI' : eachFilterConditionArr[0];
                    const isChoicesType = isChoiceType(filterName);
                    const isArray =
                        eachFilterConditionArr.length > 1 && eachFilterConditionArr[1].includes('[' || ']')
                            ? true
                            : false;
                    if (isArray === true) {
                        if (isChoicesType === true) {
                            // status:[SUCCEEDED]
                            newSearchFilter.push({
                                name: filterName,
                                operator: '=',
                                value1: '',
                                value2: '',
                                choice: convertStringArrToList(eachFilterConditionArr[1]),
                                isChoice: true
                            });
                        } else {
                            // drop_rate:[1,10]
                            newSearchFilter.push({
                                name: eachFilterConditionArr[0],
                                operator: 'between',
                                value1: JSON.parse(eachFilterConditionArr[1].trim())[0],
                                value2: JSON.parse(eachFilterConditionArr[1].trim())[1],
                                choice: [],
                                isChoice: false
                            });
                        }
                    } else {
                        newSearchFilter.push({
                            name: eachFilterConditionArr[0],
                            operator: '=',
                            value1: eachFilterConditionArr[1].trim(),
                            value2: '',
                            choice: [],
                            isChoice: false
                        });
                    }
                } else {
                    // user input: Trial id
                    return;
                }
            }
        });
        changeTableListPage(newSearchFilter);
    }

    // clear search input all value, clear all search filter
    function clearFliter(): void {
        changeTableListPage([]);
    }

    return (
        <div>
            <Stack horizontal>
                <DefaultButton text='Filter' menuProps={searchMenuProps} />
                {/* search input: store filter conditons, also, user could input filter conditions, could search */}
                <SearchBox
                    styles={{ root: { width: 530 } }}
                    placeholder='Search'
                    onChange={updateSearchText}
                    value={searchInputVal}
                    onSearch={startFilter}
                    onEscape={clearFliter}
                    onClear={clearFliter}
                />
            </Stack>
        </div>
    );
}

Search.propTypes = {
    searchFilter: PropTypes.array,
    changeSearchFilterList: PropTypes.func,
    updatePage: PropTypes.func
};

export default Search;
