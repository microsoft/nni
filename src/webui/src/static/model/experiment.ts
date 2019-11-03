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

    public async init(): Promise<void> {
        while (!this.profileField || !this.statusField) {
            await this.update();
        }
    }

    public async update(): Promise<boolean> {
        const profilePromise = axios.get(`${MANAGER_IP}/experiment`);
        const statusPromise = axios.get(`${MANAGER_IP}/check-status`);
        const [ profileResponse, statusResponse ] = await Promise.all([ profilePromise, statusPromise ]);
        let updated = false;
        if (statusResponse.status === 200) {
            updated = JSON.stringify(this.statusField) === JSON.stringify(statusResponse.data);
            this.statusField = statusResponse.data;
        }
        if (profileResponse.status === 200) {
            updated = updated || compareProfiles(this.profileField, profileResponse.data);
            this.profileField = profileResponse.data;
        }
        return updated;
    }

    get profile(): ExperimentProfile {
        if (!this.profileField) {
            throw Error('Experiment profile not initialized');
        }
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

    get multiPhase(): boolean {
        return !!(this.profile.params.multiPhase);
    }

    get status(): string {
        if (!this.statusField) {
            throw Error('Experiment status not initialized');
        }
        return this.statusField!.status;
    }

    get error(): string {
        if (!this.statusField) {
            throw Error('Experiment status not initialized');
        }
        return this.statusField!.errors[0] || '';
    }
}

export { Experiment };
