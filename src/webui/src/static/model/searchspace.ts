import { SingleAxis, MultipleAxes, TableObj } from '../interface';


function getFullName(prefix: string, name: string): string {
    return prefix ? (prefix + '/' + name) : name;
}

class NumericAxis implements SingleAxis {
    min: number = 0;
    max: number = 0;
    type: string;
    name: string;
    fullName: string;
    scale: 'log' | 'linear';
    nested = false;

    constructor(name: string, fullName: string, type: string, value: any) {
        this.name = name;
        this.fullName = fullName;
        this.type = type;
        this.scale = type.indexOf('log') !== -1 ? 'log' : 'linear';
        if (type === 'randint') {
            this.min = value[0];
            this.max = value[1] - 1;
        } else if (type.indexOf('uniform') !== -1) {
            this.min = value[0];
            this.max = value[1];
        } else if (type.indexOf('normal') !== -1) {
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
    name: string;
    fullName: string;
    scale: 'ordinal' = 'ordinal';
    domain: any[];
    nested = false;
    constructor(name: string, fullName: string, type: string, value: any) {
        this.name = name;
        this.fullName = fullName;
        this.type = type;
        this.domain = value;
    }
}

class NestedOrdinalAxis implements SingleAxis {
    type: string;
    name: string;
    fullName: string;
    scale: 'ordinal' = 'ordinal';
    domain = new Map<string, MultipleAxes>();
    nested = true;
    constructor(name: any, fullName: string, type: any, value: any) {
        this.name = name;
        this.fullName = fullName;
        this.type = type;
        for (const v of value) {
            // eslint-disable-next-line @typescript-eslint/no-use-before-define
            this.domain.set(v._name, new SearchSpace(v._name, fullName + v._name, v));
        }
    }
}

export class SearchSpace implements MultipleAxes {
    axes = new Map<string, SingleAxis>();
    name: string;
    fullName: string;

    constructor(name: string, fullName: string, searchSpaceSpec: any) {
        this.name = name;
        this.fullName = fullName;
        if (searchSpaceSpec === undefined)
            return;
        Object.entries(searchSpaceSpec).forEach((item) => {
            const key = item[0], spec = item[1] as any;
            if (spec._type === 'choice' || spec._type === 'layer_choice' || spec._type === 'input_choice') {
                // ordinal types
                if (spec._value && typeof spec._value[0] === 'object') {
                    // nested dimension
                    this.axes.set(key, new NestedOrdinalAxis(key, getFullName(fullName, key), spec._type, spec._value));
                } else {
                    this.axes.set(key, new SimpleOrdinalAxis(key, getFullName(fullName, key), spec._type, spec._value));
                }
            } else {
                this.axes.set(key, new NumericAxis(key, fullName + key, spec._type, spec._value));
            }
        });
    }

    static inferFromTrials(searchSpace: SearchSpace, trials: TableObj[]): SearchSpace {
        const newSearchSpace = new SearchSpace(searchSpace.name, searchSpace.fullName, undefined);
        for (const [k, v] of searchSpace.axes) {
            newSearchSpace.axes.set(k, v);
        }
        // Add axis inferred from trials columns
        const addingColumns = new Map<string, any[]>();
        for (const trial of trials) {
            try {
                trial.parameters(searchSpace);
            } catch (unexpectedEntries) {
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
                newSearchSpace.axes.set(key, new NumericAxis(key, key, 'uniform', [Math.min(...value), Math.max(...value)]));
            } else {
                newSearchSpace.axes.set(key, new SimpleOrdinalAxis(key, key, 'choice', new Set(value).values()));
            }
        });
        return newSearchSpace;
    }
}

export class MetricSpace implements MultipleAxes {
    axes = new Map<string, SingleAxis>();
    name = '';
    fullName = '';

    constructor(trials: TableObj[]) {
        const columns = new Map<string, any[]>();
        for (const trial of trials) {
            if (trial.acc === undefined)
                continue;
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
                // TODO: skip for now
            }
        });
    }
}
