// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as assert from 'assert';

import { getLogger, Logger } from '../common/log';
import { isAlive, withLock, withLockSync, getExperimentsInfoPath, delay } from '../common/utils';
import { ExperimentManager } from '../common/experimentManager';
import { Deferred } from 'ts-deferred';

interface CrashedInfo {
    experimentId: string;
    isCrashed: boolean;
}

interface FileInfo {
    buffer: Buffer;
    mtime: number;
}

class NNIExperimentsManager implements ExperimentManager {
    private experimentsPath: string;
    private log: Logger;
    private profileUpdateTimer: {[key: string]: any};

    constructor() {
        this.experimentsPath = getExperimentsInfoPath();
        this.log = getLogger();
        this.profileUpdateTimer = {};
    }

    public async getExperimentsInfo(): Promise<JSON> {
        const fileInfo = await this.readExperimentsInfo();
        const experimentsInformation = JSON.parse(fileInfo.buffer.toString());
        const expIdList: Array<string> = Object.keys(experimentsInformation).filter((expId) => {
            return experimentsInformation[expId]['status'] !== 'STOPPED';
        });
        const updateList: Array<CrashedInfo> = (await Promise.all(expIdList.map((expId) => {
            return this.checkCrashed(expId, experimentsInformation[expId]['pid']);
        }))).filter(crashedInfo => crashedInfo.isCrashed);
        if (updateList.length > 0){
            const result = await this.withLock(this.updateAllStatus, updateList.map(crashedInfo => crashedInfo.experimentId), fileInfo.mtime);
            if (result !== undefined) {
                return result;
            } else {
                delay(500);
                return await this.getExperimentsInfo();
            }
        } else {
            return JSON.parse(JSON.stringify(Object.keys(experimentsInformation).map(key=>experimentsInformation[key])));
        }
    }

    public setExperimentPath(newPath: string): void {
        if (newPath[0] === '~') {
            newPath = path.join(os.homedir(), newPath.slice(1));
        }
        if (!path.isAbsolute(newPath)) {
            newPath = path.resolve(newPath);
        }
        this.log.info(`Set new experiment information path: ${newPath}`);
        this.experimentsPath = newPath;
    }

    public setExperimentInfo(experimentId: string, key: string, value: any): void {
        try {
            if (this.profileUpdateTimer[key] !== undefined) {
                // if a new call with the same timerId occurs, destroy the unfinished old one
                clearTimeout(this.profileUpdateTimer[key]);
                this.profileUpdateTimer[key] = undefined;
            }
            this.withLockSync(() => {
                const experimentsInformation = JSON.parse(fs.readFileSync(this.experimentsPath).toString());
                assert(experimentId in experimentsInformation, `Experiment Manager: Experiment Id ${experimentId} not found, this should not happen`);
                experimentsInformation[experimentId][key] = value;
                fs.writeFileSync(this.experimentsPath, JSON.stringify(experimentsInformation));
            });
        } catch (err) {
            this.log.error(err);
            this.log.debug(`Experiment Manager: Retry set key value: ${experimentId} {${key}: ${value}}`);
            if (err.code === 'EEXIST') {
                this.profileUpdateTimer[key] = setTimeout(this.setExperimentInfo.bind(this), 1000, experimentId, key, value);
            }
        }
    }

    private async withLock (func: Function, ...args: any): Promise<any> {
        return withLock(func.bind(this), this.experimentsPath, {stale: 2 * 1000, retries: 100, retryWait: 100}, ...args);
    }

    private withLockSync (func: Function, ...args: any): any {
        return withLockSync(func.bind(this), this.experimentsPath, {stale: 2 * 1000}, ...args);
    }

    private async readExperimentsInfo(): Promise<FileInfo> {
        return this.withLock((path: string) => {
                const buffer: Buffer = fs.readFileSync(path);
                const mtime: number = fs.statSync(path).mtimeMs;
                return {buffer: buffer, mtime: mtime};
            },
            this.experimentsPath);
    }

    private async checkCrashed(expId: string, pid: number): Promise<CrashedInfo> {
        const alive: boolean = await isAlive(pid);
        return {experimentId: expId, isCrashed: !alive}
    }

    private updateAllStatus(updateList: Array<string>, timestamp: number): {[key: string]: any} | undefined {
        if (timestamp !== fs.statSync(this.experimentsPath).mtimeMs) {
            return;
        } else {
            const experimentsInformation = JSON.parse(fs.readFileSync(this.experimentsPath).toString());
            updateList.forEach((expId: string) => {
                if (experimentsInformation[expId]) {
                    experimentsInformation[expId]['status'] = 'STOPPED';
                } else {
                    this.log.error(`Experiment Manager: Experiment Id ${expId} not found, this should not happen`);
                }
            });
            fs.writeFileSync(this.experimentsPath, JSON.stringify(experimentsInformation));
            return experimentsInformation;
        }
    }

    public async stop(): Promise<void> {
        this.log.debug('Stopping experiment manager.');
        await this.cleanUp().catch(err=>this.log.error(err.message));
        this.log.debug('Experiment manager stopped.');
    }

    private async cleanUp(): Promise<void> {
        const deferred = new Deferred<void>();
        if (this.isUndone()) {
            this.log.debug('Experiment manager: something undone');
            setTimeout(((deferred: Deferred<void>): void => {
                if (this.isUndone()) {
                    deferred.reject(new Error('Still has undone after 5s, forced stop.'));
                } else {
                    deferred.resolve();
                }
            }).bind(this), 5 * 1000, deferred);
        } else {
            this.log.debug('Experiment manager: all clean up');
            deferred.resolve();
        }
        return deferred.promise;
    }

    private isUndone(): boolean {
        return Object.keys(this.profileUpdateTimer).filter((key: string) => {
            return this.profileUpdateTimer[key] !== undefined;
        }).length > 0;
    }
}

export { NNIExperimentsManager };
