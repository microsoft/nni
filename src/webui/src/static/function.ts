import { FinalResult, FinalType } from './interface';

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
const getFinalResult = (final: FinalResult) => {
    let acc;
    let showDefault = 0;
    if (final) {
        acc = JSON.parse(final[0].data);
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
const getFinal = (final: FinalResult) => {
    let showDefault: FinalType;
    if (final) {
        showDefault = JSON.parse(final[0].data);
        if (typeof showDefault === 'number') {
            showDefault = { default: showDefault };
        }
        return showDefault;
    } else {
        return undefined;
    }
};

export {
    convertTime, convertDuration, getFinalResult,
    getFinal
};
