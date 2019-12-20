import axios from 'axios';
import { MANAGER_IP, METRIC_GROUP_UPDATE_THRESHOLD, METRIC_GROUP_UPDATE_SIZE } from '../const';
import { MetricDataRecord, TableRecord, TrialJobInfo } from '../interface';
import { Trial } from './trial';

function groupMetricsByTrial(metrics: MetricDataRecord[]): Map<string, MetricDataRecord[]> {
    const ret = new Map<string, MetricDataRecord[]>();
    for (const metric of metrics) {
        if (ret.has(metric.trialJobId)) {
            ret.get(metric.trialJobId)!.push(metric);
        } else {
            ret.set(metric.trialJobId, [ metric ]);
        }
    }
    return ret;
}

class TrialManager {
    private trials: Map<string, Trial> = new Map<string, Trial>();
    private infoInitialized: boolean = false;
    private metricInitialized: boolean = false;
    private maxSequenceId: number = 0;
    private doingBatchUpdate: boolean = false;
    private batchUpdatedAfterReading: boolean = false;

    public async init(): Promise<void> {
        while (!this.infoInitialized || !this.metricInitialized) {
            await this.update();
        }
    }

    public async update(lastTime?: boolean): Promise<boolean> {
        const [ infoUpdated, metricUpdated ] = await Promise.all([ this.updateInfo(), this.updateMetrics(lastTime) ]);
        return infoUpdated || metricUpdated;
    }

    public getTrial(trialId: string): Trial {
        return this.trials.get(trialId)!;
    }

    public getTrials(trialIds: string[]): Trial[] {
        return trialIds.map(trialId => this.trials.get(trialId)!);
    }

    public table(trialIds: string[]): TableRecord[] {
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
        return this.filter(trial => trial.sortable).sort((trial1, trial2) => trial1.compareAccuracy(trial2)!);
    }

    public countStatus(): Map<string, number> {
        const cnt = new Map<string, number>([
            [ 'UNKNOWN', 0 ],
            [ 'WAITING', 0 ],
            [ 'RUNNING', 0 ],
            [ 'SUCCEEDED', 0 ],
            [ 'FAILED', 0 ],
            [ 'USER_CANCELED', 0 ],
            [ 'SYS_CANCELED', 0 ],
            [ 'EARLY_STOPPED', 0 ],
        ]);
        for (const trial of this.trials.values()) {
            if (trial.initialized()) {
                cnt.set(trial.info.status, cnt.get(trial.info.status)! + 1);
            }
        }
        return cnt;
    }

    private async updateInfo(): Promise<boolean> {
        const response = await axios.get(`${MANAGER_IP}/trial-jobs`);
        let updated = false;
        if (response.status === 200) {
            for (const info of response.data as TrialJobInfo[]) {
                if (this.trials.has(info.id)) {
                    updated = this.trials.get(info.id)!.updateTrialJobInfo(info) || updated;
                } else {
                    this.trials.set(info.id, new Trial(info, undefined));
                    updated = true;
                }
                this.maxSequenceId = Math.max(this.maxSequenceId, info.sequenceId);
            }
            this.infoInitialized = true;
        }
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
        const response = await axios.get(`${MANAGER_IP}/metric-data`);
        return (response.status === 200) && this.doUpdateMetrics(response.data as MetricDataRecord[], false);
    }

    private async updateLatestMetrics(): Promise<boolean> {
        const response = await axios.get(`${MANAGER_IP}/metric-data-latest`);
        return (response.status === 200) && this.doUpdateMetrics(response.data as MetricDataRecord[], true);
    }

    private async updateManyMetrics(): Promise<void> {
        if (this.doingBatchUpdate) {
            return;
        }
        this.doingBatchUpdate = true;
        for (let i = 0; i < this.maxSequenceId; i += METRIC_GROUP_UPDATE_SIZE) {
            const response = await axios.get(`${MANAGER_IP}/metric-data-range/${i}/${i + METRIC_GROUP_UPDATE_SIZE}`);
            if (response.status === 200) {
                const updated = this.doUpdateMetrics(response.data as MetricDataRecord[], false);
                this.batchUpdatedAfterReading = this.batchUpdatedAfterReading || updated;
            }
        }
        this.doingBatchUpdate = false;
    }

    private doUpdateMetrics(allMetrics: MetricDataRecord[], latestOnly: boolean): boolean {
        let updated = false;
        for (const [ trialId, metrics ] of groupMetricsByTrial(allMetrics).entries()) {
            if (this.trials.has(trialId)) {
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
