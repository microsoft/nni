import { SingleAxis, MultipleAxes, TableObj } from '../interface';
import { SUPPORTED_SEARCH_SPACE_TYPE } from '../const';
import { formatComplexTypeValue } from '../function';

function fullNameJoin(prefix: string, name: string): string {
    return prefix ? prefix + '/' + name : name;
}

class NumericAxis implements SingleAxis {
    min: number = 0;
    max: number = 0;
    type: string;
    baseName: string;
    fullName: string;
    scale: 'log' | 'linear';
    nested = false;

    constructor(baseName: string, fullName: string, type: string, value: any) {
        this.baseName = baseName;
        this.fullName = fullName;
        this.type = type;
        this.scale = type.includes('log') ? 'log' : 'linear';
        if (type === 'randint') {
            this.min = value[0];
            this.max = value[1] - 1;
        } else if (type.includes('uniform')) {
            this.min = value[0];
            this.max = value[1];
        } else if (type.includes('normal')) {
            const [mu, sigma] = [value[0], value[1]];
            this.min = mu - 4 * sigma;
            this.max = mu + 4 * sigma;
            if (this.scale === 'log') {
                this.min = Math.exp(this.min);
                this.max = Math.exp(this.max);
            }
        }
    }

    get domain(): [number, number] {
        return [this.min, this.max];
    }
}

class SimpleOrdinalAxis implements SingleAxis {
    type: string;
    baseName: string;
    fullName: string;
    scale: 'ordinal' = 'ordinal';
    domain: any[];
    nested = false;
    constructor(baseName: string, fullName: string, type: string, value: any) {
        this.baseName = baseName;
        this.fullName = fullName;
        this.type = type;
        this.domain = Array.from(value).map(formatComplexTypeValue);
    }
}

class NestedOrdinalAxis implements SingleAxis {
    type: string;
    baseName: string;
    fullName: string;
    scale: 'ordinal' = 'ordinal';
    domain = new Map<string, MultipleAxes>();
    nested = true;
    constructor(baseName: any, fullName: string, type: any, value: any) {
        this.baseName = baseName;
        this.fullName = fullName;
        this.type = type;
        for (const v of value) {
            // eslint-disable-next-line @typescript-eslint/no-use-before-define
            this.domain.set(v._name, new SearchSpace(v._name, fullNameJoin(fullName, v._name), v));
        }
    }
}

export class SearchSpace implements MultipleAxes {
    axes = new Map<string, SingleAxis>();
    baseName: string;
    fullName: string;

    constructor(baseName: string, fullName: string, searchSpaceSpec: any) {
        this.baseName = baseName;
        this.fullName = fullName;
        if (searchSpaceSpec === undefined) {
            return;
        }
        Object.entries(searchSpaceSpec).forEach(item => {
            const key = item[0],
                spec = item[1] as any;
            if (key === '_name') {
                return;
            } else if (['choice', 'layer_choice', 'input_choice'].includes(spec._type)) {
                // ordinal types
                if (spec._value && typeof spec._value[0] === 'object') {
                    // nested dimension
                    this.axes.set(
                        key,
                        new NestedOrdinalAxis(key, fullNameJoin(fullName, key), spec._type, spec._value)
                    );
                } else {
                    this.axes.set(
                        key,
                        new SimpleOrdinalAxis(key, fullNameJoin(fullName, key), spec._type, spec._value)
                    );
                }
            } else if (SUPPORTED_SEARCH_SPACE_TYPE.includes(spec._type)) {
                this.axes.set(key, new NumericAxis(key, fullName + key, spec._type, spec._value));
            }
        });
    }

    static inferFromTrials(searchSpace: SearchSpace, trials: TableObj[]): SearchSpace {
        const newSearchSpace = new SearchSpace(searchSpace.baseName, searchSpace.fullName, undefined);
        for (const [k, v] of searchSpace.axes) {
            newSearchSpace.axes.set(k, v);
        }
        // Add axis inferred from trials columns
        const addingColumns = new Map<string, any[]>();
        for (const trial of trials) {
            try {
                trial.parameters(searchSpace);
            } catch (unexpectedEntries) {
                // eslint-disable-next-line no-console
                console.warn(unexpectedEntries);
                for (const [k, v] of unexpectedEntries as Map<string, any>) {
                    const column = addingColumns.get(k);
                    if (column === undefined) {
                        addingColumns.set(k, [v]);
                    } else {
                        column.push(v);
                    }
                }
            }
        }
        addingColumns.forEach((value, key) => {
            if (value.every(v => typeof v === 'number')) {
                newSearchSpace.axes.set(
                    key,
                    new NumericAxis(key, key, 'uniform', [Math.min(...value), Math.max(...value)])
                );
            } else {
                newSearchSpace.axes.set(key, new SimpleOrdinalAxis(key, key, 'choice', new Set(value).values()));
            }
        });
        return newSearchSpace;
    }
}

export class MetricSpace implements MultipleAxes {
    axes = new Map<string, SingleAxis>();
    baseName = '';
    fullName = '';

    constructor(trials: TableObj[]) {
        const columns = new Map<string, any[]>();
        for (const trial of trials) {
            if (trial.acc === undefined) {
                continue;
            }
            // TODO: handle more than number and object
            const acc = typeof trial.acc === 'number' ? { default: trial.acc } : trial.acc;
            Object.entries(acc).forEach(item => {
                const [k, v] = item;
                const column = columns.get(k);
                if (column === undefined) {
                    columns.set(k, [v]);
                } else {
                    column.push(v);
                }
            });
        }
        columns.forEach((value, key) => {
            if (value.every(v => typeof v === 'number')) {
                this.axes.set(key, new NumericAxis(key, key, 'uniform', [Math.min(...value), Math.max(...value)]));
            } else {
                this.axes.set(key, new SimpleOrdinalAxis(key, key, 'choice', value));
            }
        });
    }
}
