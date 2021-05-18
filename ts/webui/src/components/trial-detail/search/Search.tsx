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
import SearchParameterConditions from './SearchParameterConditions';
import GeneralSearch from './GeneralSearch';
import { classNames } from './searchStyle';
import { SearchItems } from '../../../static/interface';

function Search(props): any {
    const { searchFilter, changeSearchFilterList, updatePage } = props;
    const [searchInputVal, setSearchInputVal] = useState('');

    function getSearchItems(parameterList): IContextualMenuProps {
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
                            // component: GeneralSearch
                            onRender: renderIdAndNo.bind(item)
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
                            onRender: renderParametersSearchInputs.bind(item)
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
    const searchMenuProps: IContextualMenuProps = getSearchItems(
        EXPERIMENT.isNestedExp() ? [] : Object.keys(EXPERIMENT.searchSpace)
    );

    function renderParametersSearchInputs(item: IContextualMenuItem, dismissMenu: () => void): JSX.Element {
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

    function renderIdAndNo(item: IContextualMenuItem, dismissMenu: () => void): JSX.Element {
        return (
            <GeneralSearch
                idOrTrialNo={item.text}
                searchFilter={searchFilter} // search fliter list
                changeSearchFilterList={changeSearchFilterList}
                setSearchInputVal={setSearchInputVal}
                updatePage={updatePage}
                dismiss={dismissMenu}
            />
        );
    }

    function _updateSearchText(_, newValue): void {
        setSearchInputVal(newValue);
    }

    // update TableList page
    function changeTableListPage(searchFilterList: Array<SearchItems>): void {
        changeSearchFilterList(searchFilterList);
        updatePage();
    }

    // SearchBox onSearch event: Filter based on the filter criteria entered by the user
    function startFilter(): void {
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
            let eachFilterConditionArr: string[] = [];

            if (eachFilterConditionStr.includes('Status')) {
                const splitOperator = eachFilterConditionStr.includes('≠') ? '≠' : ':';
                const filterOperator = eachFilterConditionStr.includes('≠') ? '≠' : '=';
                eachFilterConditionArr = eachFilterConditionStr.trim().split(splitOperator);
                newSearchFilter.push({
                    name: 'StatusNNI',
                    operator: filterOperator,
                    value1: eachFilterConditionArr[1],
                    value2: ''
                });
            } else {
                if (eachFilterConditionStr.includes(':')) {
                    eachFilterConditionArr = eachFilterConditionStr.trim().split(':');
                    const isArray = eachFilterConditionArr[1].includes('[' || ']')
                        ? Array.isArray(JSON.parse(eachFilterConditionArr[1]))
                        : false;
                    newSearchFilter.push({
                        name: eachFilterConditionArr[0],
                        operator: isArray ? 'between' : '=',
                        value1: isArray ? JSON.parse(eachFilterConditionArr[1])[0] : eachFilterConditionArr[1],
                        value2: isArray ? JSON.parse(eachFilterConditionArr[1])[1] : ''
                    });
                } else {
                    const operator = eachFilterConditionStr.includes('>') === true ? '>' : '<';
                    eachFilterConditionArr = eachFilterConditionStr.trim().split(operator);
                    newSearchFilter.push({
                        name: eachFilterConditionArr[0],
                        operator: operator,
                        value1: eachFilterConditionArr[1],
                        value2: ''
                    });
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
                    onChange={_updateSearchText}
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
