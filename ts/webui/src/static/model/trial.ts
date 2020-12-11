import * as JSON5 from 'json5';
import {
    MetricDataRecord,
    TrialJobInfo,
    TableObj,
    TableRecord,
    Parameters,
    FinalType,
    MultipleAxes,
    SingleAxis
} from '../interface';
import {
    getFinal,
    formatAccuracy,
    metricAccuracy,
    parseMetrics,
    isArrayType,
    isNaNorInfinity,
    formatComplexTypeValue
} from '../function';

/**
 * Get a structured representation of parameters
 * @param paramObj Parameters object
 * @param space All axes from search space (or sub search space)
 * @param prefix Current namespace (to make full name for unexpected entries)
 * @returns Parsed structured parameters and unexpected entries
 */
function inferTrialParameters(
    paramObj: object,
    space: MultipleAxes,
    prefix: string = ''
): [Map<SingleAxis, any>, Map<string, any>] {
    const parameters = new Map<SingleAxis, any>();
    const unexpectedEntries = new Map<string, any>();
    for (const [k, v] of Object.entries(paramObj)) {
        // prefix can be a good fallback when corresponding item is not found in namespace
        const axisKey = space.axes.get(k);
        if (prefix && k === '_name') continue;
        if (axisKey !== undefined) {
            if (typeof v === 'object' && v._name !== undefined && axisKey.nested) {
                // nested entry
                parameters.set(axisKey, v._name);
                const subSpace = axisKey.domain.get(v._name);
                if (subSpace !== undefined) {
                    const [subParams, subUnexpected] = inferTrialParameters(v, subSpace, prefix + k + '/');
                    subParams.forEach((v, k) => parameters.set(k, v));
                    subUnexpected.forEach((v, k) => unexpectedEntries.set(k, v));
                }
            } else {
                parameters.set(axisKey, formatComplexTypeValue(v));
            }
        } else {
            unexpectedEntries.set(prefix + k, formatComplexTypeValue(v));
        }
    }
    return [parameters, unexpectedEntries];
}

class Trial implements TableObj {
    private metricsInitialized: boolean = false;
    private infoField: TrialJobInfo | undefined;
    public intermediates: (MetricDataRecord | undefined)[] = [];
    public final: MetricDataRecord | undefined;
    private finalAcc: number | undefined;

    constructor(info?: TrialJobInfo, metrics?: MetricDataRecord[]) {
        this.infoField = info;
        if (metrics) {
            this.updateMetrics(metrics);
        }
    }

    public compareAccuracy(otherTrial: Trial): number | undefined {
        if (!this.sortable || !otherTrial.sortable) {
            return undefined;
        }
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.finalAcc! - otherTrial.finalAcc!;
    }

    get info(): TrialJobInfo {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.infoField!;
    }

    get intermediateMetrics(): MetricDataRecord[] {
        const ret: MetricDataRecord[] = [];
        for (let i = 0; i < this.intermediates.length; i++) {
            if (this.intermediates[i]) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                ret.push(this.intermediates[i]!);
            } else {
                break;
            }
        }
        return ret;
    }

    get accuracy(): number | undefined {
        return this.finalAcc;
    }

    get sortable(): boolean {
        return this.metricsInitialized && this.finalAcc !== undefined && !isNaN(this.finalAcc);
    }

    get latestAccuracy(): number | undefined {
        if (this.accuracy !== undefined) {
            return this.accuracy;
        } else if (this.intermediates.length > 0) {
            const temp = this.intermediates[this.intermediates.length - 1];
            if (temp !== undefined) {
                if (isArrayType(parseMetrics(temp.data))) {
                    return undefined;
                } else if (
                    typeof parseMetrics(temp.data) === 'object' &&
                    parseMetrics(temp.data).hasOwnProperty('default')
                ) {
                    return parseMetrics(temp.data).default;
                } else if (typeof parseMetrics(temp.data) === 'number') {
                    return parseMetrics(temp.data);
                }
            } else {
                return undefined;
            }
        } else {
            return undefined;
        }
    }
    /* table obj start */

    get tableRecord(): TableRecord {
        const endTime = this.info.endTime || new Date().getTime();
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const duration = (endTime - this.info.startTime!) / 1000;
        let accuracy;
        if (this.acc !== undefined && this.acc.default !== undefined) {
            if (typeof this.acc.default === 'number') {
                accuracy = JSON5.parse(this.acc.default);
            } else {
                accuracy = this.acc.default;
            }
        }

        return {
            key: this.info.trialJobId,
            sequenceId: this.info.sequenceId,
            id: this.info.trialJobId,
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            startTime: this.info.startTime!,
            endTime: this.info.endTime,
            duration,
            status: this.info.status,
            intermediateCount: this.intermediates.length,
            accuracy: accuracy,
            latestAccuracy: this.latestAccuracy,
            formattedLatestAccuracy: this.formatLatestAccuracy(),
            accDictionary: this.acc
        };
    }

    get key(): number {
        return this.info.sequenceId;
    }

    get sequenceId(): number {
        return this.info.sequenceId;
    }

    get id(): string {
        return this.info.trialJobId;
    }

    get duration(): number {
        const endTime = this.info.endTime || new Date().getTime();
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return (endTime - this.info.startTime!) / 1000;
    }

    get status(): string {
        return this.info.status;
    }

    get acc(): FinalType | undefined {
        if (this.info === undefined) {
            return undefined;
        }
        return getFinal(this.info.finalMetricData);
    }

    get description(): Parameters {
        const ret: Parameters = {
            parameters: {},
            intermediate: [],
            multiProgress: 1
        };
        const tempHyper = this.info.hyperParameters;
        if (tempHyper !== undefined) {
            const getPara = JSON.parse(tempHyper[tempHyper.length - 1]).parameters;
            ret.multiProgress = tempHyper.length;
            if (typeof getPara === 'string') {
                ret.parameters = JSON.parse(getPara);
            } else {
                ret.parameters = getPara;
            }
        } else {
            ret.parameters = { error: "This trial's parameters are not available." };
        }
        if (this.info.logPath !== undefined) {
            ret.logPath = this.info.logPath;
        }

        const mediate: number[] = [];
        for (const items of this.intermediateMetrics) {
            if (typeof parseMetrics(items.data) === 'object') {
                mediate.push(parseMetrics(items.data).default);
            } else {
                mediate.push(parseMetrics(items.data));
            }
        }
        ret.intermediate = mediate;
        return ret;
    }

    public parameters(axes: MultipleAxes): Map<SingleAxis, any> {
        const ret = new Map<SingleAxis, any>(Array.from(axes.axes.values()).map(k => [k, null]));
        if (this.info === undefined || this.info.hyperParameters === undefined) {
            throw ret;
        } else {
            const tempHyper = this.info.hyperParameters;
            let params = JSON.parse(tempHyper[tempHyper.length - 1]).parameters;
            if (typeof params === 'string') {
                params = JSON.parse(params);
            }
            const [updated, unexpectedEntries] = inferTrialParameters(params, axes);
            if (unexpectedEntries.size) {
                throw unexpectedEntries;
            }
            for (const [k, v] of updated) {
                ret.set(k, v);
            }
            return ret;
        }
    }

    public metrics(space: MultipleAxes): Map<SingleAxis, any> {
        // set default value: null
        const ret = new Map<SingleAxis, any>(Array.from(space.axes.values()).map(k => [k, null]));
        const unexpectedEntries = new Map<string, any>();
        if (this.acc === undefined) {
            return ret;
        }
        const acc = typeof this.acc === 'number' ? { default: this.acc } : this.acc;
        Object.entries(acc).forEach(item => {
            const [k, v] = item;
            const column = space.axes.get(k);

            if (column !== undefined) {
                ret.set(column, v);
            } else {
                unexpectedEntries.set(k, v);
            }
        });
        if (unexpectedEntries.size) {
            throw unexpectedEntries;
        }
        return ret;
    }

    get color(): string | undefined {
        return undefined;
    }

    public finalKeys(): string[] {
        if (this.acc !== undefined) {
            return Object.keys(this.acc);
        } else {
            return [];
        }
    }

    /* table obj end */

    public initialized(): boolean {
        return Boolean(this.infoField);
    }

    public updateMetrics(metrics: MetricDataRecord[]): boolean {
        // parameter `metrics` must contain all known metrics of this trial
        this.metricsInitialized = true;
        const prevMetricCnt = this.intermediates.length + (this.final ? 1 : 0);
        if (metrics.length <= prevMetricCnt) {
            return false;
        }
        for (const metric of metrics) {
            if (metric.type === 'PERIODICAL') {
                this.intermediates[metric.sequence] = metric;
            } else {
                this.final = metric;
                this.finalAcc = metricAccuracy(metric);
            }
        }
        return true;
    }

    public updateLatestMetrics(metrics: MetricDataRecord[]): boolean {
        // this method is effectively identical to `updateMetrics`, but has worse performance
        this.metricsInitialized = true;
        let updated = false;
        for (const metric of metrics) {
            if (metric.type === 'PERIODICAL') {
                updated = updated || !this.intermediates[metric.sequence];
                this.intermediates[metric.sequence] = metric;
            } else {
                updated = updated || !this.final;
                this.final = metric;
                this.finalAcc = metricAccuracy(metric);
            }
        }
        return updated;
    }

    public updateTrialJobInfo(trialJobInfo: TrialJobInfo): boolean {
        const same = this.infoField && this.infoField.status === trialJobInfo.status;
        this.infoField = trialJobInfo;
        if (trialJobInfo.finalMetricData) {
            this.final = trialJobInfo.finalMetricData[trialJobInfo.finalMetricData.length - 1];
            this.finalAcc = metricAccuracy(this.final);
        }
        return !same;
    }

    private renderNumber(val: any): string {
        if (typeof val === 'number') {
            if (isNaNorInfinity(val)) {
                return `${val}`; // show 'NaN' or 'Infinity'
            } else {
                if (this.accuracy === undefined) {
                    return `${formatAccuracy(val)} (LATEST)`;
                } else {
                    return `${formatAccuracy(val)} (FINAL)`;
                }
            }
        } else {
            // show other types, such as {tensor: {data: }}
            return JSON.stringify(val);
        }
    }

    public formatLatestAccuracy(): string {
        // TODO: this should be private
        if (this.status === 'SUCCEEDED') {
            return this.accuracy === undefined ? '--' : this.renderNumber(this.accuracy);
        } else {
            if (this.accuracy !== undefined) {
                return this.renderNumber(this.accuracy);
            } else if (this.intermediates.length === 0) {
                return '--';
            } else {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                const latest = this.intermediates[this.intermediates.length - 1]!;
                return this.renderNumber(metricAccuracy(latest));
            }
        }
    }
}

export { Trial };
