import { mergeStyleSets } from '@fluentui/react';
import { trialJobStatus } from '@static/const';
import { EXPERIMENT } from '@static/datamodel';
import { TableObj, SearchItems } from '@static/interface';

const classNames = mergeStyleSets({
    menu: {
        textAlign: 'center',
        maxWidth: 600,
        selectors: {
            '.ms-ContextualMenu-item': {
                height: 'auto'
            }
        }
    },
    item: {
        display: 'inline-block',
        width: 40,
        height: 40,
        lineHeight: 40,
        textAlign: 'center',
        verticalAlign: 'middle',
        marginBottom: 8,
        cursor: 'pointer',
        selectors: {
            '&:hover': {
                backgroundColor: '#eaeaea'
            }
        }
    },
    categoriesList: {
        margin: 0,
        padding: 0,
        listStyleType: 'none'
    },
    button: {
        width: '40%',
        margin: '2%'
    }
});

function getDropdownOptions(parameter): any {
    if (parameter === 'StatusNNI') {
        return trialJobStatus.map(item => ({
            key: item,
            text: item
        }));
    } else {
        return EXPERIMENT.searchSpace[parameter]._value.map(item => ({
            key: item.toString(),
            text: item.toString()
        }));
    }
}

// change origin data according to parameter type, string -> number
const convertParametersValue = (searchItems: SearchItems[], relation: Map<string, string>): SearchItems[] => {
    const choice: any[] = [];
    const copySearchItems = JSON.parse(JSON.stringify(searchItems));
    copySearchItems.forEach(item => {
        if (relation.get(item.name) === 'number') {
            if (item.isChoice === true) {
                item.choice.forEach(ele => {
                    choice.push(JSON.parse(ele));
                });
                item.choice = choice;
            } else {
                item.value1 = JSON.parse(item.value1);
                if (item.value2 !== '') {
                    item.value2 = JSON.parse(item.value2);
                }
            }
        }
    });

    return copySearchItems;
};
// relation: trial parameter -> type {conv_size -> number}
const getTrialsBySearchFilters = (
    arr: TableObj[],
    searchItems: SearchItems[],
    relation: Map<string, string>
): TableObj[] => {
    const que = convertParametersValue(searchItems, relation);
    // start to filter data by ['Trial id', 'Trial No.', 'Status'] [...parameters]...
    que.forEach(element => {
        if (element.name === 'Trial id') {
            arr = arr.filter(trial => trial.id.toUpperCase().includes(element.value1.toUpperCase()));
        } else if (element.name === 'Trial No.') {
            arr = arr.filter(trial => trial.sequenceId.toString() === element.value1);
        } else if (element.name === 'StatusNNI') {
            arr = searchChoiceFilter(arr, element, 'status');
        } else {
            const parameter = `space/${element.name}`;

            if (element.isChoice === true) {
                arr = searchChoiceFilter(arr, element, element.name);
            } else {
                if (element.operator === '=') {
                    arr = arr.filter(trial => trial[parameter] === element.value1);
                } else if (element.operator === '>') {
                    arr = arr.filter(trial => trial[parameter] > element.value1);
                } else if (element.operator === '<') {
                    arr = arr.filter(trial => trial[parameter] < element.value1);
                } else if (element.operator === 'between') {
                    arr = arr.filter(trial => trial[parameter] > element.value1 && trial[parameter] < element.value2);
                } else {
                    // operator is '≠'
                    arr = arr.filter(trial => trial[parameter] !== element.value1);
                }
            }
        }
    });

    return arr;
};

// isChoice = true: status and trial parameters
function findTrials(arr: TableObj[], choice: string[], filed: string): TableObj[] {
    const newResult: TableObj[] = [];
    const parameter = filed === 'status' ? 'status' : `space/${filed}`;
    arr.forEach(trial => {
        choice.forEach(item => {
            if (trial[parameter] === item) {
                newResult.push(trial);
            }
        });
    });

    return newResult;
}

function searchChoiceFilter(arr: TableObj[], element: SearchItems, field: string): TableObj[] {
    if (element.operator === '=') {
        return findTrials(arr, element.choice, field);
    } else {
        let choice;
        if (field === 'status') {
            choice = trialJobStatus.filter(index => !new Set(element.choice).has(index));
        } else {
            choice = EXPERIMENT.searchSpace[field]._value.filter(index => !new Set(element.choice).has(index));
        }
        return findTrials(arr, choice, field);
    }
}

// click Apply btn: set searchBox value now
function getSearchInputValueBySearchList(searchFilter): string {
    let str = ''; // store search input value

    searchFilter.forEach(item => {
        const filterName = item.name === 'StatusNNI' ? 'Status' : item.name;

        if (item.isChoice === false) {
            // id, No, !choice parameter
            if (item.name === 'Trial id' || item.name === 'Trial No.') {
                str = str + `${item.name}:${item.value1}; `;
            } else {
                // !choice parameter
                if (['=', '≠', '>', '<'].includes(item.operator)) {
                    str = str + `${filterName}${item.operator === '=' ? ':' : item.operator}${item.value1}; `;
                } else {
                    // between
                    str = str + `${filterName}:[${item.value1},${item.value2}]; `;
                }
            }
        } else {
            // status, choice parameter
            str = str + `${filterName}${item.operator === '=' ? ':' : '≠'}[${[...item.choice]}]; `;
        }
    });

    return str;
}

/***
 * from experiment search space
* "conv_size": {
        "_type": "choice", // is choice type
        "_value": [
            2,
            3,
            5,
            7
        ]
    },
 */
function isChoiceType(parameterName): boolean {
    // 判断是 [choice, status] 还是普通的类型
    let flag = false; // normal type

    if (parameterName === 'StatusNNI') {
        flag = true;
    }

    if (parameterName in EXPERIMENT.searchSpace) {
        flag = EXPERIMENT.searchSpace[parameterName]._type === 'choice' ? true : false;
    }

    return flag;
}

export { classNames, getDropdownOptions, getTrialsBySearchFilters, getSearchInputValueBySearchList, isChoiceType };
