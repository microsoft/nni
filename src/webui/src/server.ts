import { Server, Model } from 'miragejs';

export function makeServer(data, environment = 'test') {
    return new Server({
        environment,

        routes() {
            this.namespace = '/api/v1/nni';
            this.get('/check-status', data.checkStatus);
            this.get('/experiment', data.experiment);
            this.get('/metric-data', data.metricData);
            this.get('/trial-jobs', data.trialJobs);
            this.get('/version', () => { return 'v999.0' as any; });
        },
    })
}
