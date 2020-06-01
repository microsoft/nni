import * as JSON5 from 'json5';
import axios from 'axios';
import { MANAGER_IP } from './const';
import { MetricDataRecord, FinalType, TableObj } from './interface';

async function requestAxios(url: string) {
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
        const min = Math.floor(num / 60 % 60);
        return hour > 0 ? `${hour}h ${min}min` : `${min}min`;
    }
};

// trial's duration, accurate to seconds for example 10min 30s
const convertDuration = (num: number): string => {
    if (num < 1) {
        return '0s';
    }
    const hour = Math.floor(num / 3600);
    const minute = Math.floor(num / 60 % 60);
    const second = Math.floor(num % 60);
    const result: string[] = [];
    if (hour > 0) {
        result.push(`${hour}h`);
    }
    if (minute > 0) {
        result.push(`${minute}min`);
    }
    if (second > 0) {
        result.push(`${second}s`);
    }
    return result.join(' ');
};

function parseMetrics(metricData: string): any {
    if (metricData.includes('NaN')) {
        return JSON5.parse(JSON5.parse(metricData));
    } else {
        return JSON.parse(JSON.parse(metricData));
    }
}

const isArrayType = (list: any): boolean | undefined => {
    return Array.isArray(list);
}

// get final result value
// draw Accuracy point graph
const getFinalResult = (final?: MetricDataRecord[]): number => {
    let acc;
    let showDefault = 0;
    if (final) {
        acc = parseMetrics(final[final.length - 1].data);
        if (typeof (acc) === 'object' && !isArrayType(acc)) {
            if (acc.default) {
                showDefault = acc.default;
            }
        } else if (typeof (acc) === 'number') {
            showDefault = acc;
        } else {
            showDefault = NaN;
        }
        return showDefault;
    } else {
        return 0;
    }
};

// get final result value // acc obj
const getFinal = (final?: MetricDataRecord[]): FinalType | undefined => {
    let showDefault: FinalType;
    if (final) {
        showDefault = parseMetrics(final[final.length - 1].data);
        if (typeof showDefault === 'number') {
            if(!isNaN(showDefault)){
                showDefault = { default: showDefault };
                return showDefault;
            }
        } else if (isArrayType(showDefault)) {
            // not support final type
            return undefined;
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
                color: '#333',
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
            data: intermediateArr
        },
        series: [{
            symbolSize: 6,
            type: 'scatter',
            data: intermediateArr
        }]
    };
};

// kill job
const killJob = (key: number, id: string, status: string, updateList?: Function): void => {
    axios(`${MANAGER_IP}/trial-jobs/${id}`, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json;charset=utf-8'
        }
    })
        .then(res => {
            if (res.status === 200) {
                // TODO: use Message.txt to tooltip
                alert('Cancel the job successfully');
                // render the table
                if (updateList) {
                    updateList();  // FIXME
                }
            } else {
                alert('fail to cancel the job');
            }
        })
        .catch(error => {
            if (error.response.status === 500) {
                if (error.response.data.error) {
                    alert(123);
                    // message.error(error.response.data.error);
                } else {
                    alert(234);
                    // message.error('500 error, fail to cancel the job');
                }
            }
        });
};

const filterByStatus = (item: TableObj): boolean => {
    return item.status === 'SUCCEEDED';
};

// a waittiong trial may havn't start time 
const filterDuration = (item: TableObj): boolean => {
    return item.status !== 'WAITING';
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

export {
    convertTime, convertDuration, getFinalResult, getFinal, downFile,
    intermediateGraphOption, killJob, filterByStatus, filterDuration,
    formatAccuracy, formatTimestamp, metricAccuracy, parseMetrics,
    isArrayType, requestAxios
};
