import { AllExperimentList, SortInfo } from '../../static/interface';
import { copyAndSort } from '../../static/function';

function compareDate(date1: Date, date2: Date): boolean {
    if (date1 !== undefined && date2 !== undefined) {
        if (date1.getFullYear() === date2.getFullYear()) {
            if (date1.getMonth() === date2.getMonth()) {
                if (date1.getDate() === date2.getDate()) {
                    return true;
                }
            }
        }
    }

    return false;
}

const filterByStatusOrPlatform = (
    val: string | string[],
    type: string,
    data: AllExperimentList[]
): AllExperimentList[] => {
    
    if (typeof val === 'string' && val !== '') {
        return data.filter(temp => temp[type] === val);
    }

    if (Array.isArray(val) && val.length !== 0) {
        return data.filter(temp => val.includes(temp[type]));
    }

    return data;
};

function fillOptions(arr: string[]): any {
    const list: Array<object> = [];

    arr.map(item => {
        list.push({ key: item, text: item });
    });

    return list;
}

function getSortedSource(source: AllExperimentList[], sortInfo: SortInfo): AllExperimentList[] {
    const keepSortedSource = copyAndSort(source, sortInfo.field, sortInfo.isDescend);
    return keepSortedSource;
}

export { compareDate, filterByStatusOrPlatform, fillOptions, getSortedSource };
