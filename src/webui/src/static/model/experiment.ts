import axios from 'axios';
import { MANAGER_IP } from '../const';
import { ExperimentProfile, NNIManagerStatus } from '../interface';

function compareProfiles(profile1?: ExperimentProfile, profile2?: ExperimentProfile): boolean {
    if (!profile1 || !profile2) {
        return false;
    }
    const copy1 = Object.assign({}, profile1, { execDuration: undefined });
    const copy2 = Object.assign({}, profile2, { execDuration: undefined });
    return JSON.stringify(copy1) === JSON.stringify(copy2);
}

class Experiment {
    private profileField?: ExperimentProfile = undefined;
    private statusField?: NNIManagerStatus = undefined;
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
        const profilePromise = axios.get(`${MANAGER_IP}/experiment`);
        const statusPromise = axios.get(`${MANAGER_IP}/check-status`);
        const [profileResponse, statusResponse] = await Promise.all([profilePromise, statusPromise]);
        // const profileResponse = { data: { error: 'profileResponse error' }, status: 200 };
        // const statusResponse = { data: { error: 'statusResponse error' }, status: 200 };
        let updated = false;

        if (statusResponse.status === 200) {
            if (statusResponse.data.error !== undefined) {
                this.isStatusError = true;
                this.statusErrorMessage = statusResponse.data.error;
                updated = true;
            } else {
                updated = JSON.stringify(this.statusField) === JSON.stringify(statusResponse.data);
                this.statusField = statusResponse.data as any;
            }
        } else {
            this.isStatusError = true;
            this.statusErrorMessage = `${statusResponse.status} error`;
            updated = true;
        }

        if (profileResponse.status === 200) {
            if (profileResponse.data.error !== undefined) {
                this.isexperimentError = true;
                this.experimentErrorMessage = profileResponse.data.error;
                updated = true;
            } else {
                updated = updated || compareProfiles(this.profileField, profileResponse.data as any);
                this.profileField = profileResponse.data as any;
            }
        } else {
            this.isexperimentError = true;
            this.experimentErrorMessage = `${profileResponse.status} error`;
            updated = true;
        }
        return updated;
    }

    get profile(): ExperimentProfile {
        if (!this.profileField) {
            // throw Error('Experiment profile not initialized');
            // set initProfile to prevent page broken
            const initProfile = {
                data: {
                    "id": "", "revision": 0, "execDuration": 0,
                    "logDir": "", "nextSequenceId": 0,
                    "params": {
                        "authorName": "", "experimentName": "", "trialConcurrency": 0, "maxExecDuration": 0, "maxTrialNum": 0, "searchSpace": "null",
                        "trainingServicePlatform": "", "tuner": {
                            "builtinTunerName": "TPE",
                            "classArgs": { "optimize_mode": "" }, "checkpointDir": ""
                        },
                        "versionCheck": true, "clusterMetaData": [{ "key": "", "value": "" },
                        { "key": "", "value": "" }]
                    }, "startTime": 0, "endTime": 0
                }
            };
            this.profileField = initProfile.data as any;
        }
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return this.profileField!;
    }

    get trialConcurrency(): number {
        return this.profile.params.trialConcurrency;
    }

    get optimizeMode(): string {
        const tuner = this.profile.params.tuner;
        return (tuner && tuner.classArgs && tuner.classArgs.optimize_mode) ? tuner.classArgs.optimize_mode : 'unknown';
    }

    get trainingServicePlatform(): string {
        return this.profile.params.trainingServicePlatform;
    }

    get searchSpace(): object {
        return JSON.parse(this.profile.params.searchSpace);
    }

    get logCollectionEnabled(): boolean {
        return !!(this.profile.params.logCollection && this.profile.params.logCollection !== 'none');
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
