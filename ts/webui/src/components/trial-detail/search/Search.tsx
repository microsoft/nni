import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Stack, DefaultButton, IContextualMenuProps, IContextualMenuItem, DirectionalHint } from '@fluentui/react';
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
            />
        );
    }

    function renderIdAndNo(item: IContextualMenuItem): JSX.Element {
        return (
            <GeneralSearch
                idOrTrialNo={item.text}
                searchFilter={searchFilter} // search的数组
                changeSearchFilterList={changeSearchFilterList}
                updatePage={updatePage}
            />
        );
    }

    function _updateSearchText(ev: React.ChangeEvent<HTMLInputElement>): void {
        setSearchInputVal(ev.target.value);
    }

    function clickFuc(): void {
        // 根据 input val 来反天 searchFilter []
        const result = searchInputVal.trim().split(';');
        console.info(result);
        const copySearchFilter = JSON.parse(JSON.stringify(searchFilter));
        result.forEach(temp => {
            // conv_size = 1024, conv_size > 1024, conv_size < 1024, ≠
            // const temp = item.split(' ');
            const item = temp.trim().split(' ');
            console.info(item);
            // 先找有没有这个条件存在
            const find = copySearchFilter.filter(index => index.name === item[0]);
            if (find.length > 0) {
                // 条件存在，覆盖值
                copySearchFilter.forEach(a => {
                    if (a.name === item[0]) {
                        a.operator = item[1];
                        a.value1 = item[2];
                        a.value2 = '';
                    }
                });
            } else {
                // 不存在这个条件，直接push进去
                copySearchFilter.push({
                    name: item[0],
                    operator: item[1],
                    value1: item[2],
                    value2: ''
                });
            }
        });
        // conv_size < 7; hidden_size < 1024
        console.info(copySearchFilter);
        changeSearchFilterList(copySearchFilter);
        updatePage();
    }

    return (
        <div>
            <Stack horizontal>
                {/* 联动菜单demo */}
                <DefaultButton text='Filter' menuProps={searchMenuProps} />
                {/* 存放filter条件；用户输入filter条件，反向实现搜索 */}
                <input
                    type='text'
                    className='input-padding'
                    placeholder='xxx'
                    onChange={_updateSearchText}
                    value={searchInputVal}
                    style={{ width: 360 }}
                />
                <DefaultButton text='filter' onClick={clickFuc} />
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
