import { MANAGER_IP } from '../const';
import { ExperimentConfig, toSeconds } from '../experimentConfig';
import { ExperimentProfile, NNIManagerStatus } from '../interface';
import { requestAxios } from '../function';
import { SearchSpace } from './searchspace';

function compareProfiles(profile1?: ExperimentProfile, profile2?: ExperimentProfile): boolean {
    if (!profile1 || !profile2) {
        return false;
    }
    const copy1 = Object.assign({}, profile1, { execDuration: undefined });
    const copy2 = Object.assign({}, profile2, { execDuration: undefined });
    return JSON.stringify(copy1) === JSON.stringify(copy2);
}

const emptyProfile: ExperimentProfile = {
    params: {
        searchSpace: undefined,
        trialCommand: '',
        trialCodeDirectory: '',
        trialConcurrency: 0,
        debug: false,
        trainingService: {
            platform: ''
        }
    },
    id: '',
    execDuration: 0,
    logDir: '',
    startTime: 0,
    maxSequenceId: 0,
    revision: 0
};

class Experiment {
    private profileField?: ExperimentProfile;
    private statusField?: NNIManagerStatus = undefined;
    private isNestedExperiment: boolean = false;
    private isexperimentError: boolean = false;
    private experimentErrorMessage: string = '';
    private isStatusError: boolean = false;
    private statusErrorMessage: string = '';

    public async init(): Promise<void> {
        while (!this.profileField || !this.statusField) {
            if (this.isexperimentError) {
                return;
            }
            if (this.isStatusError) {
                return;
            }
            await this.update();
        }
    }

    public isNestedExp(): boolean {
        try {
            return !!Object.values(this.config.searchSpace).find(
                item => (item as any)._value && typeof (item as any)._value[0] == 'object'
            );
        } catch {
            return false;
        }
    }

    public experimentError(): boolean {
        return this.isexperimentError;
    }

    public statusError(): boolean {
        return this.isStatusError;
    }

    public getExperimentMessage(): string {
        return this.experimentErrorMessage;
    }

    public getStatusMessage(): string {
        return this.statusErrorMessage;
    }

    public async update(): Promise<boolean> {
        let updated = false;

        await requestAxios(`${MANAGER_IP}/experiment`)
            .then(data => {
                updated = updated || !compareProfiles(this.profileField, data);
                this.profileField = data;
            })
            .catch(error => {
                this.isexperimentError = true;
                this.experimentErrorMessage = `${error.message}`;
                updated = true;
            });

        await requestAxios(`${MANAGER_IP}/check-status`)
            .then(data => {
                updated = JSON.stringify(this.statusField) !== JSON.stringify(data);
                this.statusField = data;
            })
            .catch(error => {
                this.isStatusError = true;
                this.statusErrorMessage = `${error.message}`;
                updated = true;
            });

        return updated;
    }

    get profile(): ExperimentProfile {
        return this.profileField === undefined ? emptyProfile : this.profileField;
    }

    get config(): ExperimentConfig {
        return this.profile.params;
    }

    get maxExperimentDurationSeconds(): number {
        const value = this.config.maxExperimentDuration;
        return value === undefined ? Infinity : toSeconds(value);
    }

    get maxTrialNumber(): number {
        const value = this.config.maxTrialNumber;
        return value === undefined ? Infinity : value;
    }

    get trialConcurrency(): number {
        return this.config.trialConcurrency;
    }

    get optimizeMode(): string {
        for (const algo of [this.config.tuner, this.config.advisor, this.config.assessor]) {
            if (algo && algo.classArgs && algo.classArgs['optimizeMode']) {
                return algo.classArgs['optimizeMode'];
            }
        }
        return 'unknown';
    }

    get trainingServicePlatform(): string {
        return this.config.trainingService.platform;
    }

    get searchSpace(): object {
        return this.config.searchSpace;
    }

    get searchSpaceNew(): SearchSpace {
        // The search space derived directly from profile
        // eventually this will replace searchSpace
        return new SearchSpace('', '', this.searchSpace);
    }

    get logCollectionEnabled(): boolean {
        return false;
    }

    get status(): string {
        if (!this.statusField) {
            // throw Error('Experiment status not initialized');
            // this.statusField.status = '';
            return '';
        }
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.statusField!.status;
    }

    get error(): string {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        if (!this.statusField) {
            throw Error('Experiment status not initialized');
        }
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.statusField!.errors[0] || '';
    }
}

export { Experiment };
