import { Server, Model } from 'miragejs';
import * as mnistTfv1RunningCheckStatus from './mock/mnist-tfv1-running.json';

export function makeServer(environment = 'test', data = 'mnist-tfv1-running') {
    let fullData;
    if (data === 'mnist-tfv1-running') {
        fullData = (mnistTfv1RunningCheckStatus as any).default;
    }
    return new Server({
        environment,

        routes() {
            this.namespace = '/api/v1/nni';
            this.get('/check-status', fullData.checkStatus);
            this.get('/experiment', fullData.experiment);
            this.get('/metric-data', fullData.metricData);
            this.get('/trial-jobs', fullData.trialJobs);
            this.get('/version', 'v999.0' as any);
        },
    })
}
