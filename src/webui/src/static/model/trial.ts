import { MetricDataRecord, TrialJobInfo, TableObj, TableRecord, Parameters, FinalType, StructuredItem, MultipleAxes } from '../interface';
import { getFinal, formatAccuracy, metricAccuracy, parseMetrics, isArrayType } from '../function';
import { stringify } from 'json5';

class Trial implements TableObj {
    private metricsInitialized: boolean = false;
    private infoField: TrialJobInfo | undefined;
    private intermediates: (MetricDataRecord | undefined)[] = [];
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
                } else if (typeof parseMetrics(temp.data) === 'object' && parseMetrics(temp.data).hasOwnProperty('default')) {
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

        return {
            key: this.info.id,
            sequenceId: this.info.sequenceId,
            id: this.info.id,
            jobId: this.info.jobId,
            parameterId: this.info.parameterId,
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            startTime: this.info.startTime!,
            endTime: this.info.endTime,
            duration,
            status: this.info.status,
            intermediateCount: this.intermediates.length,
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            accuracy: this.acc !== undefined ? JSON.parse(this.acc!.default) : undefined,
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
        return this.info.id;
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
            ret.parameters = { error: 'This trial\'s parameters are not available.' };
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

    public parameters(axes: StructuredItem[]): Map<StructuredItem, any> {
        const ret = new Map<StructuredItem, any>();
        const flatten = (source: object,
                         namespace: StructuredItem[],
                         prefix: string,
                         ignoreName: boolean = false) => {
            const namespaceMap = new Map<string, StructuredItem>();
            for (const item of namespace) {
                namespaceMap.set(item.name, item);
            }
            Object.entries(source).forEach(item => {
                const [k, v] = item;
                // prefix can be a good fallback when corresponding item is not found in namespace
                const axisKey = namespaceMap.get(k) || { name: prefix + k, fullName: prefix + k, parent: undefined, children: [] };
                if (ignoreName && k === '_name')
                    return;
                if (typeof v === 'object' && (v as any)._name !== undefined) {
                    // nested entry
                    ret.set(axisKey, (v as any)._name);
                    flatten(v, axisKey.children, prefix + k + '/', true);
                } else {
                    ret.set(axisKey, v);
                }
            });
            return ret;
        };
        const tempHyper = this.info.hyperParameters;
        if (tempHyper === undefined) {
            const key = { name: 'error', fullName: 'error', parent: undefined, children: [] };
            return new Map([[key, 'This trial\'s parameters are not available.']]);
        } else {
            let getPara = JSON.parse(tempHyper[tempHyper.length - 1]).parameters;
            if (typeof getPara === 'string') {
                getPara = JSON.parse(getPara);
            }
            flatten(getPara as object, axes, '');
            return ret;
        }
    }

    public metrics(): Map<StructuredItem, any> {
        const ret = new Map<StructuredItem, any>();
        if (this.acc === undefined) {
            return ret;
        }
        for (const [k, v] of Object.entries(this.acc)) {
            ret.set({name: k, fullName: k, parent: undefined, children: []}, v);
        }
        return ret;
    }

    get color(): string | undefined {
        return undefined;
    }

    public finalKeys(): string[] {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return Object.keys(this.acc!);
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
        const same = (this.infoField && this.infoField.status === trialJobInfo.status);
        this.infoField = trialJobInfo;
        if (trialJobInfo.finalMetricData) {
            this.final = trialJobInfo.finalMetricData[trialJobInfo.finalMetricData.length - 1];
            this.finalAcc = metricAccuracy(this.final);
        }
        return !same;
    }

    public formatLatestAccuracy(): string {  // TODO: this should be private
        if (this.accuracy !== undefined) {
            if (isNaN(this.accuracy)) {
                return this.accuracy.toString();
            } else {
                return `${formatAccuracy(this.accuracy)} (FINAL)`;
            }
        } else if (this.intermediates.length === 0) {
            return '--';
        } else {
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            const latest = this.intermediates[this.intermediates.length - 1]!;
            if (isNaN(metricAccuracy(latest))) {
                return 'NaN';
            } else {
                return `${formatAccuracy(metricAccuracy(latest))} (LATEST)`;
            }
        }
    }
}

export { Trial };
