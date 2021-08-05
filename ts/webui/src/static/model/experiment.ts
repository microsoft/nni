import { MANAGER_IP } from '../const';
import { ExperimentConfig, toSeconds } from '../experimentConfig';
import { ExperimentProfile, ExperimentMetadata, NNIManagerStatus } from '../interface';
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

const emptyMetadata: ExperimentMetadata = {
    id: '',
    port: 0,
    startTime: '',
    endTime: '',
    status: '',
    platform: '',
    experimentName: '',
    tag: [],
    pid: 0,
    webuiUrl: [],
    logDir: '',
    prefixUrl: null
};

class Experiment {
    private profileField?: ExperimentProfile;
    private metadataField?: ExperimentMetadata = undefined;
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

        await Promise.all([requestAxios(`${MANAGER_IP}/experiment`), requestAxios(`${MANAGER_IP}/experiment-metadata`)])
            .then(([profile, metadata]) => {
                updated ||= !compareProfiles(this.profileField, profile);
                this.profileField = profile;

                if (JSON.stringify(this.metadataField) !== JSON.stringify(metadata)) {
                    this.metadataField = metadata;
                }
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

    get metadata(): ExperimentMetadata {
        return this.metadataField === undefined ? emptyMetadata : this.metadataField;
    }

    get config(): ExperimentConfig {
        return this.profile.params;
    }

    get maxExperimentDurationSeconds(): number {
        const value = this.config.maxExperimentDuration || (this.config as any).maxExecDuration;
        return value === undefined ? Infinity : toSeconds(value);
    }

    get maxTrialNumber(): number {
        const value = this.config.maxTrialNumber || (this.config as any).maxTrialNum;
        return value === undefined ? Infinity : value;
    }

    get trialConcurrency(): number {
        return this.config.trialConcurrency;
    }

    get optimizeMode(): string {
        for (const algo of [this.config.tuner, this.config.advisor, this.config.assessor]) {
            if (algo && algo.classArgs && algo.classArgs['optimize_mode']) {
                return algo.classArgs['optimize_mode'];
            }
        }
        return 'unknown';
    }

    get trainingServicePlatform(): string {
        if (Array.isArray(this.config.trainingService)) {
            return 'hybrid';
        } else if (this.config.trainingService) {
            return this.config.trainingService.platform;
        } else {
            return (this.config as any).trainingServicePlatform;
        }
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
