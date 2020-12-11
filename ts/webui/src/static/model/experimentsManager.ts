import { MANAGER_IP } from '../const';
import { AllExperimentList } from '../interface';
import { requestAxios } from '../function';

class ExperimentsManager {
    private experimentList: AllExperimentList[] = [];
    private platform: string[] = [];
    private errorMessage: string = '';

    public getExperimentList(): AllExperimentList[] {
        return this.experimentList;
    }

    public getPlatformList(): string[] {
        return this.platform;
    }

    public getExpErrorMessage(): string {
        return this.errorMessage;
    }

    public async init(): Promise<void> {
        await requestAxios(`${MANAGER_IP}/experiments-info`)
            .then(data => {
                const platforms: Set<string> = new Set();
                for (const item of data) {
                    if (item.port !== undefined) {
                        if (typeof item.port === 'string') {
                            item.port = JSON.parse(item.port);
                        }
                    }
                    platforms.add(item.platform);
                }
                this.experimentList = data;
                this.platform = Array.from(platforms);
            })
            .catch(error => {
                this.errorMessage = error.message;
            });
    }
}

export { ExperimentsManager };
