import { MANAGER_IP } from '../const';
import { AllExperimentList } from '../interface';
import { requestAxios } from '../function';

class ExperimentsManager {
    private experimentList: AllExperimentList[] = [];

    public getExperimentList(): AllExperimentList[] {
        return this.experimentList;
    }

    public async init(): Promise<void> {
        await requestAxios(`${MANAGER_IP}/experiments-info`).then(data => {
            for (const item of data) {
                if (typeof item.port === 'string') {
                    item.port = JSON.parse(item.port);
                }
            }
            this.experimentList = data;
        })
        .catch(error => {
            return [] as AllExperimentList[];
        });
    }
}

export { ExperimentsManager };
