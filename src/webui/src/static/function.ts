import axios from 'axios';
import { message } from 'antd';
import { MANAGER_IP } from './const';
import { FinalResult, FinalType, TableObj } from './interface';

const convertTime = (num: number) => {
    if (num % 3600 === 0) {
        return num / 3600 + 'h';
    } else {
        const hour = Math.floor(num / 3600);
        const min = Math.floor(num / 60 % 60);
        return hour > 0 ? `${hour}h ${min}min` : `${min}min`;
    }
};

// trial's duration, accurate to seconds for example 10min 30s
const convertDuration = (num: number) => {
    const hour = Math.floor(num / 3600);
    const min = Math.floor(num / 60 % 60);
    const second = Math.floor(num % 60);
    const result = hour > 0 ? `${hour} h ${min} min ${second}s` : `${min} min ${second}s`;
    if (hour <= 0 && min === 0 && second !== 0) {
        return `${second}s`;
    } else if (hour === 0 && min !== 0 && second === 0) {
        return `${min}min`;
    } else if (hour === 0 && min !== 0 && second !== 0) {
        return `${min}min ${second}s`;
    } else {
        return result;
    }
};

// get final result value
// draw Accuracy point graph
const getFinalResult = (final: Array<FinalResult>) => {
    let acc;
    let showDefault = 0;
    if (final) {
        acc = JSON.parse(final[final.length - 1].data);
        if (typeof (acc) === 'object') {
            if (acc.default) {
                showDefault = acc.default;
            }
        } else {
            showDefault = acc;
        }
        return showDefault;
    } else {
        return 0;
    }
};

// get final result value // acc obj 
const getFinal = (final: Array<FinalResult>) => {
    let showDefault: FinalType;
    if (final) {
        showDefault = JSON.parse(final[final.length - 1].data);
        if (typeof showDefault === 'number') {
            showDefault = { default: showDefault };
        }
        return showDefault;
    } else {
        return undefined;
    }
};

// detail page table intermediate button
const intermediateGraphOption = (intermediateArr: number[], id: string) => {
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
            name: 'Trial',
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
const killJob = (key: number, id: string, status: string, updateList: Function) => {
    axios(`${MANAGER_IP}/trial-jobs/${id}`, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json;charset=utf-8'
        }
    })
        .then(res => {
            if (res.status === 200) {
                message.destroy();
                message.success('Cancel the job successfully');
                // render the table
                updateList();
            } else {
                message.error('fail to cancel the job');
            }
        })
        .catch(error => {
            if (error.response.status === 500) {
                if (error.response.data.error) {
                    message.error(error.response.data.error);
                } else {
                    message.error('500 error, fail to cancel the job');
                }
            }
        });
};

const filterByStatus = (item: TableObj) => {
    return item.status === 'SUCCEEDED';
};

// a waittiong trial may havn't start time 
const filterDuration = (item: TableObj) => {
    return item.status !== 'WAITING';
};

export {
    convertTime, convertDuration, getFinalResult, getFinal, 
    intermediateGraphOption, killJob, filterByStatus, filterDuration
};
