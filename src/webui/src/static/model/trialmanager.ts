import axios from 'axios';
import { MANAGER_IP } from '../const';
import { MetricDataRecord, TableRecord, TrialJobInfo } from '../interface';
import { Trial } from './trial';

class TrialManager {
    private trials: Map<string, Trial> = new Map<string, Trial>();
    private infoInitialized: boolean = false;
    private metricInitialized: boolean = false;

    public async init(): Promise<void> {
        while (!this.infoInitialized || !this.metricInitialized) {
            await this.update();
        }
    }

    public async update(): Promise<boolean> {
        const infoPromise = axios.get(`${MANAGER_IP}/trial-jobs`);
        const metricPromise = axios.get(`${MANAGER_IP}/metric-data`);
        const [ infoResponse, metricResponse ] = await Promise.all([ infoPromise, metricPromise ]);
        let updated = false;

        if (infoResponse.status === 200) {
            for (const info of infoResponse.data as TrialJobInfo[]) {
                if (this.trials.has(info.id)) {
                    updated = this.trials.get(info.id)!.updateTrialJobInfo(info) || updated;
                } else {
                    this.trials.set(info.id, new Trial(info, undefined));
                    updated = true;
                }
            }
            this.infoInitialized = true;
        }

        if (metricResponse.status === 200) {
            const allMetrics = groupMetricsByTrial(metricResponse.data as MetricDataRecord[]);
            for (const [ trialId, metrics ] of allMetrics.entries()) {
                if (this.trials.has(trialId)) {
                    updated = this.trials.get(trialId)!.updateMetrics(metrics) || updated;
                } else {
                    this.trials.set(trialId, new Trial(undefined, metrics));
                    updated = true;
                }
            }
            this.metricInitialized = true;
        }

        return updated;
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
}

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

export { TrialManager };
