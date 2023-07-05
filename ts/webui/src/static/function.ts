import * as JSON5 from 'json5';
import axios from 'axios';
import { IContextualMenuProps } from '@fluentui/react';
import { RETIARIIPARAMETERS } from './const';
import { EXPERIMENT } from './datamodel';
import { MetricDataRecord, FinalType, TensorboardTaskInfo } from './interface';

function getPrefix(): string | undefined {
    const pathName = window.location.pathname;
    let newPathName = pathName;
    const pathArr: string[] = ['/oview', '/detail', '/experiment'];
    pathArr.forEach(item => {
        if (pathName.endsWith(item)) {
            newPathName = pathName.replace(item, '');
        }
    });
    let result = newPathName === '' || newPathName === '/' ? undefined : newPathName;
    if (result !== undefined) {
        if (result.endsWith('/')) {
            result = result.slice(0, result.length - 1);
        }
    }
    return result;
}

async function requestAxios(url: string): Promise<any> {
    const response = await axios.get(url);
    if (response.status === 200) {
        if (response.data.error !== undefined) {
            throw new Error(`API ${url} ${response.data.error}`);
        } else {
            return response.data as any;
        }
    } else {
        throw new Error(`API ${url} ${response.status} error`);
    }
}

const convertTime = (num: number): string => {
    if (num <= 0) {
        return '0';
    }
    if (num % 3600 === 0) {
        return num / 3600 + 'h';
    } else {
        const hour = Math.floor(num / 3600);
        const min = Math.floor((num / 60) % 60);
        return hour > 0 ? `${hour}h ${min}min` : `${min}min`;
    }
};

// trial's duration, accurate to seconds for example 10min 30s
const convertDuration = (seconds: number): string => {
    let str = '';

    const d = Math.floor(seconds / (24 * 3600));
    if (d > 0) {
        str += `${d}d `;
    }
    seconds -= d * 24 * 3600;

    const h = Math.floor(seconds / 3600);
    if (h > 0) {
        str += `${h}h `;
    }
    seconds -= h * 3600;

    const m = Math.floor(seconds / 60);
    if (m > 0) {
        str += `${m}m `;
    }
    seconds -= m * 60;

    // don't show `0s`
    if (Math.floor(seconds) > 0) {
        str += `${Math.floor(seconds)}s`;
    }
    return str ? str : '0s';
};

const formatTimeStyle = (seconds: number): string => {
    // 1d2h3m4s
    let str = '';

    const d = Math.floor(seconds / (24 * 3600));
    if (d > 0) {
        str += `<b>${d}</b><span>d</span>`;
    }
    seconds -= d * 24 * 3600;

    const h = Math.floor(seconds / 3600);
    if (h > 0) {
        str += `<b>${h}</b><span>h</span>`;
    }
    seconds -= h * 3600;

    const m = Math.floor(seconds / 60);
    if (m > 0) {
        str += `<b>${m}</b><span>m</span>`;
    }
    seconds -= m * 60;

    // don't show `0s`
    if (Math.floor(seconds) > 0) {
        str += `<b>${Math.floor(seconds)}</b><span>s</span>`;
    }
    return str;
};
// according the unit(d,h,m) to convert duration
function convertTimeAsUnit(unit: string, value: number): number {
    let divisor = 1;
    if (unit === 'h') {
        divisor = 3600;
    } else if (unit === 'm') {
        divisor = 60;
    } else {
        divisor = 24 * 3600;
    }
    return value / divisor;
}
function parseMetrics(metricData: string): any {
    if (metricData.includes('NaN') || metricData.includes('Infinity')) {
        return JSON5.parse(JSON5.parse(metricData));
    } else {
        return JSON.parse(JSON.parse(metricData));
    }
}

const isArrayType = (list: any): boolean | undefined => {
    return Array.isArray(list);
};

// get final result value
// draw Accuracy point graph
const getFinalResult = (final?: MetricDataRecord[]): number => {
    let acc;
    let showDefault = 0;
    if (final) {
        acc = parseMetrics(final[final.length - 1].data);
        if (typeof acc === 'object' && !isArrayType(acc)) {
            if (acc.default) {
                showDefault = acc.default;
            }
        } else if (typeof acc === 'number') {
            showDefault = acc;
        } else {
            showDefault = NaN;
        }
        return showDefault;
    } else {
        return 0;
    }
};

function isNaNorInfinity(val: number): boolean {
    return Object.is(val, NaN) || Object.is(val, Infinity);
}

// get final result value // acc obj
const getFinal = (final?: MetricDataRecord[]): FinalType | undefined => {
    let showDefault: FinalType;
    if (final) {
        showDefault = parseMetrics(final[final.length - 1].data);
        if (typeof showDefault === 'number') {
            if (!isNaNorInfinity(showDefault)) {
                return { default: showDefault };
            }
        } else if (isArrayType(showDefault)) {
            // not support final type
            return undefined;
            // eslint-disable-next-line no-prototype-builtins
        } else if (typeof showDefault === 'object' && showDefault.hasOwnProperty('default')) {
            return showDefault;
        }
    } else {
        return undefined;
    }
};

// detail page table intermediate button
const intermediateGraphOption = (intermediateArr: number[], id: string): any => {
    const sequence: number[] = [];
    const lengthInter = intermediateArr.length;
    for (let i = 1; i <= lengthInter; i++) {
        sequence.push(i);
    }
    return {
        title: {
            text: id,
            left: 'center',
            textStyle: {
                fontSize: 16,
                color: '#333'
            }
        },
        tooltip: {
            trigger: 'item'
        },
        xAxis: {
            // name: '#Intermediate result',
            data: sequence
        },
        yAxis: {
            name: 'Default metric',
            type: 'value',
            data: intermediateArr,
            scale: true
        },
        series: [
            {
                symbolSize: 6,
                type: 'scatter',
                data: intermediateArr
            }
        ]
    };
};

const downFile = (content: string, fileName: string): void => {
    const aTag = document.createElement('a');
    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
    const file = new Blob([content], { type: 'application/json' });
    aTag.download = fileName;
    aTag.href = URL.createObjectURL(file);
    aTag.click();
    if (!isEdge) {
        URL.revokeObjectURL(aTag.href);
    }
    if (navigator.userAgent.indexOf('Firefox') > -1) {
        const downTag = document.createElement('a');
        downTag.addEventListener('click', function () {
            downTag.download = fileName;
            downTag.href = URL.createObjectURL(file);
        });
        const eventMouse = document.createEvent('MouseEvents');
        eventMouse.initEvent('click', false, false);
        downTag.dispatchEvent(eventMouse);
    }
};

// function formatTimestamp(timestamp?: number, placeholder?: string = 'N/A'): string {
function formatTimestamp(timestamp?: number, placeholder?: string): string {
    if (placeholder === undefined) {
        placeholder = 'N/A';
    }
    return timestamp ? new Date(timestamp).toLocaleString('en-US') : placeholder;
}

function expformatTimestamp(timestamp: number | string): string {
    if (typeof timestamp === 'number') {
        return new Date(timestamp).toLocaleString('en-US');
    } else {
        return 'N/A';
    }
}

function metricAccuracy(metric: MetricDataRecord): number {
    const data = parseMetrics(metric.data);
    // return typeof data === 'number' ? data : NaN;
    if (typeof data === 'number') {
        return data;
    } else {
        return data.default;
    }
}

function formatAccuracy(accuracy: number): string {
    // TODO: how to format NaN?
    return accuracy.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
}

function formatComplexTypeValue(value: any): string | number {
    if (['number', 'string'].includes(typeof value)) {
        return value;
    } else {
        // for hpo experiment: search space choice value is None, and it shows null
        return String(value);
    }
}

function isManagerExperimentPage(): boolean {
    return location.pathname.indexOf('experiment') === -1 ? false : true;
}

function caclMonacoEditorHeight(height): number {
    // [Search space 64px] + [marginBottom 18px] +
    // button[height: 32px, marginTop: 45px, marginBottom: 7px]
    // panel own padding-bottom: 20px;
    return height - 186;
}

function copyAndSort<T>(items: T[], columnKey: string, isSortedDescending?: boolean): any {
    const key = columnKey as keyof T;
    return items.slice(0).sort(function (a: T, b: T): any {
        if (
            a[key] === undefined ||
            Object.is(a[key], NaN) ||
            Object.is(a[key], Infinity) ||
            Object.is(a[key], -Infinity) ||
            typeof a[key] === 'object'
        ) {
            return 1;
        }
        if (
            b[key] === undefined ||
            Object.is(b[key], NaN) ||
            Object.is(b[key], Infinity) ||
            Object.is(b[key], -Infinity) ||
            typeof b[key] === 'object'
        ) {
            return -1;
        }
        return (isSortedDescending ? a[key] < b[key] : a[key] > b[key]) ? 1 : -1;
    });
}

function disableTensorboard(selectedRowIds: string[], queryTensorboardList: TensorboardTaskInfo[]): boolean {
    let flag = true;

    if (selectedRowIds.length !== 0) {
        flag = false;
    }

    if (selectedRowIds.length === 0 && queryTensorboardList.length !== 0) {
        flag = false;
    }

    return flag;
}

function getTensorboardMenu(
    queryTensorboardList: TensorboardTaskInfo[],
    stopFunc,
    seeDetailFunc
): IContextualMenuProps {
    const result: Array<object> = [];
    if (queryTensorboardList.length !== 0) {
        result.push({
            key: 'delete',
            text: 'Stop all tensorBoard',
            className: 'clearAll',
            onClick: stopFunc
        });
        queryTensorboardList.forEach(item => {
            result.push({
                key: item.id,
                text: `${item.id}`,
                className: `CommandBarButton-${item.status}`,
                onClick: (): void => seeDetailFunc(item)
            });
        });
    }
    const tensorboardMenu: IContextualMenuProps = {
        items: result.reverse() as any
    };

    return tensorboardMenu;
}

// search space type map list: now get type from search space
const parametersType = (): Map<string, string> => {
    const parametersTypeMap = new Map();
    const trialParameterlist = Object.keys(EXPERIMENT.searchSpace);

    trialParameterlist.forEach(item => {
        parametersTypeMap.set(item, typeof EXPERIMENT.searchSpace[item]._value[0]);
    });

    return parametersTypeMap;
};

// retiarii experiment parameters is in field `mutation_summary`
const reformatRetiariiParameter = (parameters: any): {} => {
    return RETIARIIPARAMETERS in parameters ? parameters[RETIARIIPARAMETERS] : parameters;
};

function _inferColumnTitle(columnKey: string): string {
    if (columnKey === 'sequenceId') {
        return 'Trial No.';
    } else if (columnKey === 'id') {
        return 'ID';
    } else if (columnKey === 'intermediateCount') {
        return 'Intermediate results (#)';
    } else if (columnKey === 'message') {
        return 'Message';
    } else if (columnKey.startsWith('space/')) {
        return columnKey.split('/', 2)[1] + ' (space)';
    } else if (columnKey === 'latestAccuracy') {
        return 'Default metric'; // to align with the original design
    } else if (columnKey.startsWith('metric/')) {
        return columnKey.split('/', 2)[1] + ' (metric)';
    } else if (columnKey.startsWith('_')) {
        return columnKey;
    } else {
        // camel case to verbose form
        const withSpace = columnKey.replace(/[A-Z]/g, letter => ` ${letter.toLowerCase()}`);
        return withSpace.charAt(0).toUpperCase() + withSpace.slice(1);
    }
}

const getIntermediateAllKeys = (intermediateDialogTrial: any): string[] => {
    let intermediateAllKeysList: string[] = [];
    if (intermediateDialogTrial!.intermediates !== undefined && intermediateDialogTrial!.intermediates[0]) {
        const parsedMetric = parseMetrics(intermediateDialogTrial!.intermediates[0].data);
        if (parsedMetric !== undefined && typeof parsedMetric === 'object') {
            const allIntermediateKeys: string[] = [];
            // just add type=number keys
            for (const key in parsedMetric) {
                if (typeof parsedMetric[key] === 'number') {
                    allIntermediateKeys.push(key);
                }
            }
            intermediateAllKeysList = allIntermediateKeys;
        }
    }

    if (intermediateAllKeysList.includes('default') && intermediateAllKeysList[0] !== 'default') {
        intermediateAllKeysList = intermediateAllKeysList.filter(item => item !== 'default');
        intermediateAllKeysList.unshift('default');
    }

    return intermediateAllKeysList;
};

export {
    getPrefix,
    convertTime,
    convertDuration,
    convertTimeAsUnit,
    getFinalResult,
    getFinal,
    downFile,
    intermediateGraphOption,
    formatAccuracy,
    formatTimestamp,
    formatTimeStyle,
    expformatTimestamp,
    metricAccuracy,
    parseMetrics,
    isArrayType,
    requestAxios,
    isNaNorInfinity,
    formatComplexTypeValue,
    isManagerExperimentPage,
    caclMonacoEditorHeight,
    copyAndSort,
    disableTensorboard,
    getTensorboardMenu,
    parametersType,
    reformatRetiariiParameter,
    getIntermediateAllKeys,
    _inferColumnTitle
};
