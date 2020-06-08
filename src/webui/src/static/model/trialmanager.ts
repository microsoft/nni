import { MANAGER_IP, METRIC_GROUP_UPDATE_THRESHOLD, METRIC_GROUP_UPDATE_SIZE } from '../const';
import { MetricDataRecord, TableRecord, TrialJobInfo } from '../interface';
import { Trial } from './trial';
import { requestAxios } from '../function';

function groupMetricsByTrial(metrics: MetricDataRecord[]): Map<string, MetricDataRecord[]> {
    const ret = new Map<string, MetricDataRecord[]>();
    for (const metric of metrics) {
        const trialId = `${metric.trialJobId}-${metric.parameterId}`;
        metric.trialId = trialId;
        if (ret.has(trialId)) {
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            ret.get(trialId)!.push(metric);
        } else {
            ret.set(trialId, [metric]);
        }
    }
    // to compatiable with multi-trial in same job, fix offset of sequence
    ret.forEach((trialMetrics) => {
        let minSequenceNumber = Number.POSITIVE_INFINITY;
        trialMetrics.map((item) => {
            if (item.sequence < minSequenceNumber && item.type !== "FINAL") {
                minSequenceNumber = item.sequence;
            }
        });
        trialMetrics.map((item) => {
            if (item.type !== "FINAL") {
                item.sequence -= minSequenceNumber;
            }
        });
    });
    return ret;
}

class TrialManager {
    private trials: Map<string, Trial> = new Map<string, Trial>();
    private infoInitialized: boolean = false;
    private metricInitialized: boolean = false;
    private maxSequenceId: number = 0;
    private doingBatchUpdate: boolean = false;
    private batchUpdatedAfterReading: boolean = false;
    private isJobListError: boolean = false; // trial-jobs api error filed
    private jobErrorMessage: string = ''; // trial-jobs error message
    private isMetricdataError: boolean = false; // metric-data api error filed
    private MetricdataErrorMessage: string = ''; // metric-data error message
    private isLatestMetricdataError: boolean = false; // metric-data-latest api error filed
    private latestMetricdataErrorMessage: string = ''; // metric-data-latest error message
    private isMetricdataRangeError: boolean = false; // metric-data-range api error filed
    private metricdataRangeErrorMessage: string = ''; // metric-data-latest error message

    public async init(): Promise<void> {
        while (!this.infoInitialized || !this.metricInitialized) {
            if (this.isMetricdataError) {
                return;
            }
            await this.update();
        }
    }

    public async update(lastTime?: boolean): Promise<boolean> {
        const [infoUpdated, metricUpdated] = await Promise.all([this.updateInfo(), this.updateMetrics(lastTime)]);
        return infoUpdated || metricUpdated;
    }

    public getTrial(trialId: string): Trial {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.trials.get(trialId)!;
    }

    public getTrials(trialIds: string[]): Trial[] {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return trialIds.map(trialId => this.trials.get(trialId)!);
    }

    public table(trialIds: string[]): TableRecord[] {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return trialIds.map(trialId => this.trials.get(trialId)!.tableRecord);
    }

    public toArray(): Trial[] {
        const trials = Array.from(this.trials.values()).filter(trial => trial.initialized());
        return trials.sort((trial1, trial2) => trial1.sequenceId - trial2.sequenceId);
    }

    public filter(callback: (trial: Trial) => boolean): Trial[] {
        const trials = Array.from(this.trials.values()).filter(trial => trial.initialized() && callback(trial));
        return trials.sort((trial1, trial2) => trial1.sequenceId - trial2.sequenceId);
    }

    public succeededTrials(): Trial[] {
        return this.filter(trial => trial.status === 'SUCCEEDED');
    }

    public sort(): Trial[] {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.filter(trial => trial.sortable).sort((trial1, trial2) => trial1.compareAccuracy(trial2)!);
    }

    public countStatus(): Map<string, number> {
        const cnt = new Map<string, number>([
            ['UNKNOWN', 0],
            ['WAITING', 0],
            ['RUNNING', 0],
            ['SUCCEEDED', 0],
            ['FAILED', 0],
            ['USER_CANCELED', 0],
            ['SYS_CANCELED', 0],
            ['EARLY_STOPPED', 0],
        ]);
        for (const trial of this.trials.values()) {
            if (trial.initialized()) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                cnt.set(trial.info.status, cnt.get(trial.info.status)! + 1);
            }
        }
        return cnt;
    }

    public static expandJobsToTrials(jobs: TrialJobInfo[]): TrialJobInfo[] {
        const trials: TrialJobInfo[] = [];

        for (const jobInfo of jobs as TrialJobInfo[]) {
            if (jobInfo.hyperParameters) {
                let trial: TrialJobInfo | undefined;
                let lastTrial: TrialJobInfo | undefined;
                for (let i = 0; i < jobInfo.hyperParameters.length; i++) {
                    const hyperParameters = jobInfo.hyperParameters[i]
                    const hpObject = JSON.parse(hyperParameters);
                    const parameterId = hpObject["parameter_id"];
                    trial = {
                        id: `${jobInfo.id}-${parameterId}`,
                        jobId: jobInfo.id,
                        parameterId: parameterId,
                        sequenceId: parameterId,
                        status: "SUCCEEDED",
                        startTime: jobInfo.startTime,
                        endTime: jobInfo.startTime,
                        hyperParameters: [hyperParameters],
                        logPath: jobInfo.logPath,
                        stderrPath: jobInfo.stderrPath,
                    };
                    if (jobInfo.finalMetricData) {
                        for (const metricData of jobInfo.finalMetricData) {
                            if (metricData.parameterId == parameterId) {
                                trial.finalMetricData = [metricData];
                                trial.endTime = metricData.timestamp;
                                break;
                            }
                        }
                    }
                    if (lastTrial) {
                        trial.startTime = lastTrial.endTime;
                    } else {
                        trial.startTime = jobInfo.startTime;
                    }
                    lastTrial = trial;
                    trials.push(trial);
                }
                if (lastTrial !== undefined) {
                    lastTrial.status = jobInfo.status;
                    lastTrial.endTime = jobInfo.endTime;
                }
            } else {
                trials.push(jobInfo);
            }
        }
        return trials;
    }

    // if this.jobListError = true, show trial error message [/trial-jobs]
    public jobListError(): boolean {
        return this.isJobListError;
    }

    // trial error message's content [/trial-jobs]
    public getJobErrorMessage(): string {
        return this.jobErrorMessage;
    }

    // [/metric-data]
    public MetricDataError(): boolean {
        return this.isMetricdataError;
    }

    // [/metric-data]
    public getMetricDataErrorMessage(): string {
        return this.MetricdataErrorMessage;
    }

    // [/metric-data-latest]
    public latestMetricDataError(): boolean {
        return this.isLatestMetricdataError;
    }

    // [/metric-data-latest]
    public getLatestMetricDataErrorMessage(): string {
        return this.latestMetricdataErrorMessage;
    }

    public metricDataRangeError(): boolean {
        return this.isMetricdataRangeError;
    }

    public metricDataRangeErrorMessage(): string {
        return this.metricdataRangeErrorMessage;
    }

    private async updateInfo(): Promise<boolean> {

        let updated = false;
        requestAxios(`${MANAGER_IP}/trial-jobs`)
            .then(data => {
                const newTrials = TrialManager.expandJobsToTrials(data as any);
                for (const trialInfo of newTrials as TrialJobInfo[]) {
                    if (this.trials.has(trialInfo.id)) {
                        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                        updated = this.trials.get(trialInfo.id)!.updateTrialJobInfo(trialInfo) || updated;
                    } else {
                        this.trials.set(trialInfo.id, new Trial(trialInfo, undefined));
                        updated = true;
                    }
                    this.maxSequenceId = Math.max(this.maxSequenceId, trialInfo.sequenceId);
                }
                this.infoInitialized = true;
            })
            .catch(error => {
                this.isJobListError = true;
                this.jobErrorMessage = error.message;
                this.infoInitialized = true;
                updated = true;
            });

            return updated;
    }

    private async updateMetrics(lastTime?: boolean): Promise<boolean> {
        if (this.trials.size < METRIC_GROUP_UPDATE_THRESHOLD || lastTime) {
            return await this.updateAllMetrics();
        } else {
            this.updateManyMetrics();
            const ret = (await this.updateLatestMetrics()) || this.batchUpdatedAfterReading;
            this.batchUpdatedAfterReading = false;
            return ret;
        }
    }

    private async updateAllMetrics(): Promise<boolean> {
        return requestAxios(`${MANAGER_IP}/metric-data`)
            .then(data => this.doUpdateMetrics(data as any, false))
            .catch(error => {
                this.isMetricdataError = true;
                this.MetricdataErrorMessage = `${error.message}`;
                this.doUpdateMetrics([], false);
                return true;
            });
    }

    private async updateLatestMetrics(): Promise<boolean> {
        return requestAxios(`${MANAGER_IP}/metric-data-latest`)
            .then(data => this.doUpdateMetrics(data as any, true))
            .catch(error => {
                this.isLatestMetricdataError = true;
                this.latestMetricdataErrorMessage = `${error.message}`;
                this.doUpdateMetrics([], true);
                return true;
            });
    }

    private async updateManyMetrics(): Promise<void> {
        if (this.doingBatchUpdate) {
            return;
        }
        this.doingBatchUpdate = true;
        for (let i = 0; i < this.maxSequenceId && this.isMetricdataRangeError === false; i += METRIC_GROUP_UPDATE_SIZE) {
            requestAxios(`${MANAGER_IP}/metric-data-range/${i}/${i + METRIC_GROUP_UPDATE_SIZE}`)
                .then(data => {
                    const updated = this.doUpdateMetrics(data as any, false);
                    this.batchUpdatedAfterReading = this.batchUpdatedAfterReading || updated;
                })
                .catch(error => {
                    this.isMetricdataRangeError = true;
                    this.metricdataRangeErrorMessage = `${error.message}`;
                });
        }
        this.doingBatchUpdate = false;
    }

    private doUpdateMetrics(allMetrics: MetricDataRecord[], latestOnly: boolean): boolean {
        let updated = false;
        for (const [trialId, metrics] of groupMetricsByTrial(allMetrics).entries()) {
            if (this.trials.has(trialId)) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                const trial = this.trials.get(trialId)!;
                updated = (latestOnly ? trial.updateLatestMetrics(metrics) : trial.updateMetrics(metrics)) || updated;
            } else {
                this.trials.set(trialId, new Trial(undefined, metrics));
                updated = true;
            }
        }
        this.metricInitialized = true;
        return updated;
    }
}

export { TrialManager };
