import { MetricDataRecord, TrialJobInfo, TableRecord, FinalType, MultipleAxes, SingleAxis } from '../interface';
import {
    getFinal,
    formatAccuracy,
    metricAccuracy,
    parseMetrics,
    isArrayType,
    isNaNorInfinity,
    formatComplexTypeValue,
    reformatRetiariiParameter
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
    const latestedParamObj: object = reformatRetiariiParameter(paramObj);
    const parameters = new Map<SingleAxis, any>();
    const unexpectedEntries = new Map<string, any>();
    for (const [k, v] of Object.entries(latestedParamObj)) {
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

class Trial {
    private metricsInitialized: boolean = false;
    private infoField: TrialJobInfo | undefined;
    public accuracy: number | undefined; // trial default metric val: number value or undefined
    public intermediates: (MetricDataRecord | undefined)[] = [];

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
        return this.accuracy! - otherTrial.accuracy!;
    }

    get info(): TrialJobInfo {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.infoField!;
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

    get parameter(): object {
        return JSON.parse(this.info.hyperParameters![0]).parameters;
    }

    // return dict final result: {default: xxx...}
    get acc(): FinalType | undefined {
        if (this.info === undefined) {
            return undefined;
        }
        return getFinal(this.info.finalMetricData);
    }

    public parameters(axes: MultipleAxes): Map<SingleAxis, any> {
        const ret = new Map<SingleAxis, any>(Array.from(axes.axes.values()).map(k => [k, null]));
        if (this.info === undefined || this.info.hyperParameters === undefined) {
            throw ret;
        } else {
            let params = JSON.parse(this.info.hyperParameters[0]).parameters;
            if (typeof params === 'string') {
                params = JSON.parse(params);
            }
            // for hpo experiment: search space choice value is None, and it shows null
            for (const [key, value] of Object.entries(params)) {
                if (Object.is(null, value)) {
                    params[key] = 'null';
                }
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

    get sortable(): boolean {
        return this.metricsInitialized && this.accuracy !== undefined && isFinite(this.accuracy);
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
                    // eslint-disable-next-line no-prototype-builtins
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

    get accuracyNumberTypeDictKeys(): string[] {
        let accuracyTypeList: string[] = [];

        if (this.acc !== undefined) {
            for (const [item, value] of Object.entries(this.acc)) {
                if (typeof value === 'number') {
                    accuracyTypeList.push(item);
                }
            }
        } else {
            accuracyTypeList = ['default'];
        }

        return accuracyTypeList;
    }

    /* table obj start */

    get tableRecord(): TableRecord {
        const endTime = this.info.endTime || new Date().getTime();
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const duration = (endTime - this.info.startTime!) / 1000;

        return {
            _key: this.info.trialJobId,
            sequenceId: this.info.sequenceId,
            id: this.info.trialJobId,
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            startTime: this.info.startTime!,
            endTime: this.info.endTime,
            duration,
            status: this.info.status,
            message: this.info.message ?? '--',
            intermediateCount: this.intermediates.length,
            latestAccuracy: this.latestAccuracy,
            _formattedLatestAccuracy: this.formatLatestAccuracy()
        };
    }

    public metrics(space: MultipleAxes): Map<SingleAxis, any> {
        // set default value: null
        const ret = new Map<SingleAxis, any>(Array.from(space.axes.values()).map(k => [k, null]));
        const unexpectedEntries = new Map<string, any>();
        if (this.acc === undefined) {
            return ret;
        }
        Object.entries(this.acc).forEach(item => {
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

    /* table obj end */

    public initialized(): boolean {
        return Boolean(this.infoField);
    }

    public updateMetrics(metrics: MetricDataRecord[]): boolean {
        // parameter `metrics` must contain all known metrics of this trial
        this.metricsInitialized = true;
        const prevMetricCnt = this.intermediates.length + (this.accuracy ? 1 : 0);
        if (metrics.length <= prevMetricCnt) {
            return false;
        }
        for (const metric of metrics) {
            if (metric.type === 'PERIODICAL') {
                this.intermediates[metric.sequence] = metric;
            } else {
                this.accuracy = metricAccuracy(metric);
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
                updated = updated || !this.accuracy;
                this.accuracy = metricAccuracy(metric);
            }
        }
        return updated;
    }

    public updateTrialJobInfo(trialJobInfo: TrialJobInfo): boolean {
        const same = this.infoField && this.infoField.status === trialJobInfo.status;
        this.infoField = trialJobInfo;
        if (trialJobInfo.finalMetricData) {
            this.accuracy = metricAccuracy(trialJobInfo.finalMetricData[0]);
        }
        return !same;
    }

    /**
     *
     * @param val trial latest accuracy
     * @returns 0.9(FINAL) or 0.9(LATEST)
     * NaN or Infinity
     * string object such as: '{tensor: {data}}'
     *
     */
    private formatLatestAccuracyToString(val: any): string {
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

    /**
     *
     * @param val trial latest accuracy
     * @returns 0.9(FINAL) or 0.9(LATEST)
     * NaN or Infinity
     * string object such as: '{tensor: {data}}'
     * +1 describe type undefined: --
     *
     */
    private formatLatestAccuracy(): string {
        if (this.status === 'SUCCEEDED') {
            return this.accuracy === undefined ? '--' : this.formatLatestAccuracyToString(this.accuracy);
        } else {
            if (this.accuracy !== undefined) {
                return this.formatLatestAccuracyToString(this.accuracy);
            } else if (this.intermediates.length === 0) {
                return '--';
            } else {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                const latest = this.intermediates[this.intermediates.length - 1]!;
                return this.formatLatestAccuracyToString(metricAccuracy(latest));
            }
        }
    }
}

export { Trial };
