import { MetricDataRecord, TrialJobInfo, TableObj, TableRecord, Parameters, FinalType } from '../interface'; // eslint-disable-line no-unused-vars
import { getFinal, formatAccuracy, metricAccuracy, parseMetrics } from '../function';

class Trial implements TableObj {
    private metricsInitialized: boolean = false;
    private infoField: TrialJobInfo | undefined;
    private intermediates: (MetricDataRecord | undefined)[] = [ ];
    private final: MetricDataRecord | undefined;
    private finalAcc: number | undefined;

    constructor(info?: TrialJobInfo, metrics?: MetricDataRecord[]) {
        this.infoField = info;
        if (metrics) {
            this.updateMetrics(metrics);
        }
    }

    public compareAccuracy(otherTrial: Trial): number | undefined {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
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
        const ret: MetricDataRecord[] = [ ];
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
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.metricsInitialized && this.finalAcc !== undefined && !isNaN(this.finalAcc);
    }

    get latestAccuracy(): number | undefined {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        if (this.accuracy !== undefined) {
            return this.accuracy;
        } else if (this.intermediates.length > 0) {
            // TODO: support intermeidate result is dict
            const temp = this.intermediates[this.intermediates.length - 1];
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            if (temp !== undefined) {
                return parseMetrics(temp.data);
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
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            startTime: this.info.startTime!,
            endTime: this.info.endTime,
            duration,
            status: this.info.status,
            intermediateCount: this.intermediates.length,
            accuracy: this.finalAcc,
            latestAccuracy: this.latestAccuracy,
            formattedLatestAccuracy: this.formatLatestAccuracy(),
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
            parameters: { },
            intermediate: [ ],
            multiProgress: 1
        };
        const tempHyper = this.info.hyperParameters;
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
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
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        if (this.info.logPath !== undefined) {
            ret.logPath = this.info.logPath;
        }

        const mediate: number[] = [ ];
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

    get color(): string | undefined {
        return undefined;
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
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                updated = updated || !this.intermediates[metric.sequence];
                this.intermediates[metric.sequence] = metric;
            } else {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
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
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return !same;
    }

    public formatLatestAccuracy(): string {  // TODO: this should be private
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        if (this.accuracy !== undefined) {
            return `${formatAccuracy(this.accuracy)} (FINAL)`;
        } else if (this.intermediates.length === 0) {
            return '--';
        } else {
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            const latest = this.intermediates[this.intermediates.length - 1]!;
            return `${formatAccuracy(metricAccuracy(latest))} (LATEST)`;
        }
    }
}

export { Trial };
