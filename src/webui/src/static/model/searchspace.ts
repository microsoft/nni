import { SingleAxis, MultipleAxes, TableObj, StructuredItem } from '../interface';
import { axisLeft } from 'd3';

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
        if (searchSpaceSpec === undefined)
            return;
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

    static inferFromTrials(searchSpace: SearchSpace, trials: TableObj[]): SearchSpace {
        const newSearchSpace = new SearchSpace(undefined);
        for (const [k, v] of searchSpace.axes) {
            newSearchSpace.axes.set(k, v);
        }
        // Add axis inferred from trials columns
        const addingColumns = new Map<string, any[]>();
        const axes = newSearchSpace.getAxesTree();
        for (const trial of trials) {
            for (const [k, v] of trial.parameters(axes)) {
                if (k.parent !== undefined) {
                    // axis with parent must exist, because all added columns
                    // have undefined as their parent
                    continue;
                }
                if (axes.indexOf(k) !== -1)
                    continue;
                const column = addingColumns.get(k.fullName);
                if (column === undefined) {
                    addingColumns.set(k.fullName, [v]);
                } else {
                    column.push(v);
                }
            }
        }
        addingColumns.forEach((value, key) => {
            if (value.every(v => typeof v === 'number')) {
                newSearchSpace.axes.set(key, new NumericAxis('uniform', [Math.min(...value), Math.max(...value)]));
            } else {
                newSearchSpace.axes.set(key, new SimpleOrdinalAxis('choice', new Set(value).values()));
            }
        });
        return newSearchSpace;
    }

    public getAllAxes(): Map<StructuredItem, SingleAxis> {
        // this will expand all nested axes
        const ret = new Map<StructuredItem, SingleAxis>();
        const addSearchSpace = (searchSpace: MultipleAxes, parentKey: StructuredItem | undefined, prefix: string = '') => {
            searchSpace.axes.forEach((axis, k) => {
                const key = { name: k, fullName: prefix + k, children: [], parent: parentKey };
                if (parentKey !== undefined) {
                    parentKey.children.push(key);
                }
                if (axis instanceof NestedOrdinalAxis) {
                    ret.set(key, new SimpleOrdinalAxis(axis.type, axis.domain.keys()));
                    for (const [name, subSearchSpace] of axis.domain) {
                        addSearchSpace(subSearchSpace, key, prefix + name + '/');
                    }
                } else {
                    ret.set(key, axis);
                }
            });
        };
        addSearchSpace(this, undefined);
        return ret;
    }

    public getAxesTree(): StructuredItem[] {
        const allAxes = this.getAllAxes();
        const rootItems: StructuredItem[] = [];
        for (const item of allAxes.keys()) {
            if (item.parent === undefined) {
                rootItems.push(item);
            }
        }
        return rootItems;
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

    public getAllAxes(): Map<StructuredItem, SingleAxis> {
        const ret = new Map<StructuredItem, SingleAxis>();
        for (const [k, v] of this.axes) {
            ret.set({name: k, fullName: k, parent: undefined, children: []}, v);
        }
        return ret;
    }

    public getAxesTree(): StructuredItem[] {
        return Array.from(this.axes.keys()).map(s => ({name: s, fullName: s, parent: undefined, children: []}));
    }
}
