import { MANAGER_IP } from '../const';
import { AllExperimentList } from '../interface';
import { requestAxios } from '../function';

class ExperimentsManager {

    private experimentList: AllExperimentList[] = [];

    public getExperimentList(): AllExperimentList[] {
        console.info('333'); // eslint-disable-line
        console.info(this.experimentList); // eslint-disable-line
        return this.experimentList;
    }

    public async init(): Promise<void> {
        await requestAxios(`${MANAGER_IP}/experiments-info`)
            .then(data => {
                this.experimentList = data;
            });
            // .catch(error => {
            //     return [] as AllExperimentList[];
            // });

    }
}

export { ExperimentsManager };
