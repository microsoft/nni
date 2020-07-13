import { SingleAxis, MultipleAxes, TableObj } from '../interface';

class NumericAxis implements SingleAxis {
    min: number = 0;
    max: number = 0;
    type: string;
    scale: 'log' | 'linear';

    constructor(type: string, value: any) {
        this.type = type;
        this.scale = type.indexOf('log') !== -1 ? 'log' : 'linear';
        if (type === 'randint') {
            this.min = value[0];
            this.max = value[1] - 1;
        } else if (type.indexOf('uniform') !== -1) {
            this.min = value[0];
            this.max = value[1];
        } else if (type.indexOf('normal') !== -1) {
            this.min = -Infinity;
            this.max = Infinity;
        }
    }

    get domain(): [number, number] {
        return [this.min, this.max];
    }
}

class SimpleOrdinalAxis implements SingleAxis {
    type: string;
    scale: 'ordinal' = 'ordinal';
    domain: any[];
    constructor(type: string, value: any) {
        this.type = type;
        this.domain = value;
    }
}

class NestedOrdinalAxis implements SingleAxis {
    type: string;
    scale: 'ordinal' = 'ordinal';
    domain = new Map<string, MultipleAxes>();
    constructor(type: any, value: any) {
        this.type = type;
        for (const v of value) {
            // eslint-disable-next-line @typescript-eslint/no-use-before-define
            this.domain.set(v._name, new SearchSpace(v));
        }
    }
}

export class SearchSpace implements MultipleAxes {
    axes = new Map<string, SingleAxis>();

    constructor(searchSpaceSpec: any) {
        Object.entries(searchSpaceSpec).forEach((item) => {
            const key = item[0], spec = item[1] as any;
            if (spec._type === 'choice' || spec._type === 'layer_choice' || spec._type === 'input_choice') {
                // ordinal types
                if (spec._value && typeof spec._value[0] === 'object') {
                    // nested dimension
                    this.axes.set(key, new NestedOrdinalAxis(spec._type, spec._value));
                } else {
                    this.axes.set(key, new SimpleOrdinalAxis(spec._type, spec._value));
                }
            } else {
                this.axes.set(key, new NumericAxis(spec._type, spec._value));
            }
        });
    }

    public getAllAxes(): Map<string, SingleAxis> {
        // this will expand all nested axes
        const ret = new Map<string, SingleAxis>();
        const addSearchSpace = (searchSpace: MultipleAxes, prefix: string = '') => {
            searchSpace.axes.forEach((axis, k) => {
                if (axis instanceof NestedOrdinalAxis) {
                    ret.set(prefix + k, new SimpleOrdinalAxis(axis.type, axis.domain.keys()));
                    for (const [name, subSearchSpace] of axis.domain) {
                        addSearchSpace(subSearchSpace, prefix + name + '/');
                    }
                } else {
                    ret.set(prefix + k, axis);
                }
            });
        };
        addSearchSpace(this);
        return ret;
    }

    public updateWithTrials(trials: TableObj[]) {
        const allAxes = this.getAllAxes();
        const addingColumns = new Map<string, any[]>();
        for (const trial of trials) {
            Object.entries(trial.parameters()).forEach(item => {
                const [k, v] = item;
                if (allAxes.has(k))
                    return;
                const column = addingColumns.get(k);
                if (column === undefined) {
                    addingColumns.set(k, [v]);
                } else {
                    column.push(v);
                }
            });
        }
        addingColumns.forEach((value, key) => {
            if (value.every(v => typeof v === 'number')) {
                this.axes.set(key, new NumericAxis('uniform', [Math.min(...value), Math.max(...value)]));
            } else {
                this.axes.set(key, new SimpleOrdinalAxis('choice', new Set(value).values()));
            }
        });
    }
}

export class MetricSpace implements MultipleAxes {
    axes = new Map<string, SingleAxis>();

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
                this.axes.set(key, new NumericAxis('uniform', [Math.min(...value), Math.max(...value)]));
            } else {
                // TODO: skip for now
            }
        });
    }

    public getAllAxes(): Map<string, SingleAxis> {
        return this.axes;
    }
}
