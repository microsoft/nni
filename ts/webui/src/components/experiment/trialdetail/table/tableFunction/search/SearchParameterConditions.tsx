import React, { useState, useContext } from 'react';
import { Stack, PrimaryButton, Dropdown, IDropdownOption } from '@fluentui/react';
import { EXPERIMENT } from '@static/datamodel';
import { getDropdownOptions, getSearchInputValueBySearchList } from './searchFunction';
import { gap10 } from '@components/fluent/ChildrenGap';
import { AppContext } from '@/App';
import { SearchItems } from '@static/interface';

/***
 * This file is for filtering trial parameters and trial status
 * filter status
 * rules: (1) click filter -> click `Status` -> choose you wanted status, and click apply
 * rules: (2) you could input the rule in the search input
 * Status:[FAILED]; Status:[FAILED,SUCCEEDED]; Status≠[FAILED]; Status≠[FAILED,SUCCEEDED];
 *
 * filter parameters
 * parameters have many types, such as int, string
 * int: dropout_rate:[0.1,0.2]; // 0.1 < dropout_rate value < 0.2
 *      dropout_rate>0.5; // dropout_rate value > 0.5
 *      dropout_rate<0.6; // dropout_rate value < 0.6
 *
 * string: conv_size:[2]; // conv_size value is 2
 *         conv_size:2; // conv_size value is 2(more simple)
 *         conv_size:[2,3]; // conv_size value is 2 or 3
 *         conv_size≠[2,3]; // conv_size value is not 2 or 3
 *         conv_size≠[2]; // conv_size value is not 2
 *
 * wrong write:
 *         conv_size≠2; // wrong write, please don't input this!
 *         hidden_size≠1024  // please don't input this format!
 */

interface SearchParameterConditionsProps {
    parameter: string;
    searchFilter: SearchItems[];
    dismiss: () => void;
    setSearchInputVal: (a: string) => void;
    changeSearchFilterList: (a: SearchItems[]) => void;
    updatePage: () => void;
}

function SearchParameterConditions(props: SearchParameterConditionsProps): any {
    const { parameter, searchFilter, dismiss, changeSearchFilterList, setSearchInputVal } = props;
    const { updateDetailPage } = useContext(AppContext);
    const isChoiceTypeSearchFilter = parameter === 'StatusNNI' || EXPERIMENT.searchSpace[parameter]._type === 'choice';
    const operatorList = isChoiceTypeSearchFilter ? ['=', '≠'] : ['between', '>', '<', '=', '≠'];

    const initValueList = getInitVal();
    const [operatorVal, setOperatorVal] = useState(initValueList[0] as string);
    const [firstInputVal, setFirstInputVal] = useState(initValueList[1] as string);
    const [secondInputVal, setSecondInputVal] = useState(initValueList[2] as string);
    // status or choice parameter dropdown selected value list
    const [choiceList, setChoiceList] = useState(initValueList[3] as string[]);

    function getInitVal(): Array<string | string[]> {
        // push value: operator, firstInputVal(value1), secondInputVal(value2), choiceValue
        const str: Array<string | string[]> = [];

        if (searchFilter.length > 0) {
            const filterElement = searchFilter.find(ele => ele.name === parameter);
            if (filterElement !== undefined) {
                str.push(
                    filterElement.operator,
                    filterElement.value1.toString(),
                    filterElement.value2.toString(),
                    filterElement.choice.toString().split(',')
                );
            } else {
                // set init value
                str.push(`${isChoiceTypeSearchFilter ? '=' : 'between'}`, '', '', [] as string[]);
            }
        } else {
            str.push(`${isChoiceTypeSearchFilter ? '=' : 'between'}`, '', '', [] as string[]);
        }

        return str;
    }

    function updateOperatorDropdown(_event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void {
        if (item !== undefined) {
            setOperatorVal(item.key.toString());
        }
    }

    // get [status | parameters that type is choice] list
    function updateChoiceDropdown(_event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void {
        if (item !== undefined) {
            const result = item.selected
                ? [...choiceList, item.key as string]
                : choiceList.filter(key => key !== item.key);
            setChoiceList(result);
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
            return EXPERIMENT.searchSpace[parameter]._value[1].toString();
        }

        return secondInputVal as string;
    }

    // click Apply button
    function startFilterTrials(): void {
        if (isChoiceTypeSearchFilter === false) {
            if (firstInputVal === '') {
                alert('Please input related value!');
                return;
            }
        }

        if (firstInputVal.match(/[a-zA-Z]/) || secondInputVal.match(/[a-zA-Z]/)) {
            alert('Please input a number!');
            return;
        }

        let newSearchFilters = JSON.parse(JSON.stringify(searchFilter));
        const find = newSearchFilters.filter(ele => ele.name === parameter);

        if (find.length > 0) {
            // if user clear all selected options, will clear this filter condition on the searchFilter list
            // eg: conv_size -> choiceList = [], searchFilter will remove (name === 'conv_size')
            if ((isChoiceTypeSearchFilter && choiceList.length !== 0) || isChoiceTypeSearchFilter === false) {
                newSearchFilters.forEach(item => {
                    if (item.name === parameter) {
                        item.operator = operatorVal;
                        item.value1 = firstInputVal;
                        item.value2 = getSecondInputVal();
                        item.choice = choiceList;
                        item.isChoice = isChoiceTypeSearchFilter ? true : false;
                    }
                });
            } else {
                newSearchFilters = newSearchFilters.filter(item => item.name !== parameter);
            }
        } else {
            if ((isChoiceTypeSearchFilter && choiceList.length !== 0) || isChoiceTypeSearchFilter === false) {
                newSearchFilters.push({
                    name: parameter,
                    operator: operatorVal,
                    value1: firstInputVal,
                    value2: getSecondInputVal(),
                    choice: choiceList,
                    isChoice: isChoiceTypeSearchFilter ? true : false
                });
            }
        }

        setSearchInputVal(getSearchInputValueBySearchList(newSearchFilters));
        changeSearchFilterList(newSearchFilters);
        updateDetailPage();
        dismiss(); // close menu
    }

    return (
        // for trial parameters & Status
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
            {isChoiceTypeSearchFilter ? (
                <Dropdown
                    // selectedKeys:[] multiy, selectedKey: string
                    selectedKeys={choiceList}
                    multiSelect
                    options={getDropdownOptions(parameter)}
                    onChange={updateChoiceDropdown}
                    className='btn-vertical-middle'
                    styles={{ root: { width: 190 } }}
                />
            ) : (
                <React.Fragment>
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
                </React.Fragment>
            )}
            <PrimaryButton text='Apply' className='btn-vertical-middle' onClick={startFilterTrials} />
        </Stack>
    );
}

export default SearchParameterConditions;
