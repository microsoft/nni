import { Server, Model } from 'miragejs';
import * as mnistTfv1RunningCheckStatus from './mock/mnist-tfv1-running/check-status.json';
import * as mnistTfv1RunningExperiment from './mock/mnist-tfv1-running/experiment.json';
import * as mnistTfv1RunningMetricData from './mock/mnist-tfv1-running/metric-data.json';
import * as mnistTfv1RunningTrialJobs from './mock/mnist-tfv1-running/trial-jobs.json';

export function makeServer(environment = 'test') {
    return new Server({
        environment,

        routes() {
            this.namespace = '/api/v1/nni';
            this.get('/check-status', () => {
                return (mnistTfv1RunningCheckStatus as any).default;
            });
            this.get('/experiment', () => {
                return (mnistTfv1RunningExperiment as any).default;
            });
            this.get('/metric-data', () => {
                return (mnistTfv1RunningMetricData as any).default;
            });
            this.get('/trial-jobs', () => {
                return (mnistTfv1RunningTrialJobs as any).default;
            });
            this.get('/version', 'v999.0' as any);
        },
    })
}
