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

function Search(props): any {
    const { searchFilter, changeSearchFilterList, updatePage } = props;
    const [searchInputVal, setSearchInputVal] = useState('');
    function getSearchItems(parameterList): IContextualMenuProps {
        const result: Array<object> = [];
        parameterList.unshift('StatusNNI');
        ['Trial id', 'Trial No.'].forEach(item => {
            result.push({
                key: item,
                text: item,
                subMenuProps: {
                    items: [
                        {
                            key: item,
                            text: item,
                            // component: GernalSearch
                            onRender: renderIdAndNo.bind(item)
                        }
                    ]
                }
            });
        });

        parameterList.forEach(item => {
            result.push({
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
            items: result as any
        };

        return filterMenu;
    }
    // 规避 nested 实验，nested 实验不支持进行超参搜索
    const searchMenuProps: IContextualMenuProps = getSearchItems(
        EXPERIMENT.isNestedExp() ? [] : Object.keys(EXPERIMENT.searchSpace)
    );

    function renderParametersSearchInputs(item: IContextualMenuItem): JSX.Element {
        return (
            <SearchParameterConditions
                parameter={item.text}
                searchFilter={searchFilter} // search的数组
                changeSearchFilterList={changeSearchFilterList}
                updatePage={updatePage}
                setSearchInputVal={setSearchInputVal}
                key={item.id}
            />
        );
    }

    function renderIdAndNo(item: IContextualMenuItem): JSX.Element {
        return (
            <GeneralSearch
                idOrTrialNo={item.text}
                searchFilter={searchFilter} // search的数组
                changeSearchFilterList={changeSearchFilterList}
                setSearchInputVal={setSearchInputVal}
                updatePage={updatePage}
            />
        );
    }

    function _updateSearchText(_, newValue): void {
        setSearchInputVal(newValue);
    }

    // 根据用户自己填入的筛选条件来进行筛选
    function startFilter(): void {
        // 根据 input val 来反填 searchFilter []
        const result = searchInputVal.trim().split(';');
        let id = 0;
        if (result.includes('')) {
            // delete '' in filter list
            result.splice(
                result.findIndex(item => item === ''),
                1
            );
        }
        const newSearchFilter: any = [];
        result.forEach(temp => {
            let item;
            if (temp.includes('Status')) {
                const splitOperator = temp.includes('≠') ? '≠' : ':';
                const filterOperator = temp.includes('≠') ? '≠' : '=';
                item = temp.trim().split(splitOperator);
                newSearchFilter.push({
                    name: 'StatusNNI',
                    id: ++id,
                    operator: filterOperator,
                    value1: item[1],
                    value2: ''
                });
            } else {
                if (temp.includes(':')) {
                    item = temp.trim().split(':');
                    const isArray = item[1].includes('[' || ']') ? Array.isArray(JSON.parse(item[1])) : false;
                    newSearchFilter.push({
                        id: ++id,
                        name: item[0],
                        operator: isArray ? 'between' : '=',
                        value1: isArray ? JSON.parse(item[1])[0] : item[1],
                        value2: isArray ? JSON.parse(item[1])[1] : ''
                    });
                } else {
                    const operator = temp.includes('>') === true ? '>' : '<';
                    item = temp.trim().split(operator);
                    newSearchFilter.push({
                        id: ++id,
                        name: item[0],
                        operator: operator,
                        value1: item[1],
                        value2: ''
                    });
                }
            }
        });
        newSearchFilter.forEach(element => {
            console.info(element);
        });
        changeSearchFilterList(newSearchFilter);
        updatePage();
    }

    function clearFliter(): void {
        changeSearchFilterList([]);
        updatePage();
    }
    return (
        <div>
            <Stack horizontal>
                <DefaultButton text='Filter' menuProps={searchMenuProps} />
                {/* 存放filter条件；用户输入filter条件，反向实现搜索 */}
                <SearchBox
                    styles={{ root: { width: 400 } }}
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
