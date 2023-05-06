// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import fs from 'fs';
import * as timersPromises from 'timers/promises';

import { Deferred } from 'ts-deferred';

import { getLogger, Logger } from 'common/log';
import globals from 'common/globals';
import { isAlive } from 'common/utils';
import { withLock, withLockNoWait } from './utils';

const logger: Logger = getLogger('experiments_manager');

interface CrashedInfo {
    experimentId: string;
    isCrashed: boolean;
}

interface FileInfo {
    buffer: Buffer;
    mtime: number;
}

export class ExperimentsManager {
    private profileUpdateTimer: Record<string, NodeJS.Timeout | undefined> = {};

    constructor() {
        globals.shutdown.register('experiments_manager', this.cleanUp.bind(this));
    }

    public async getExperimentsInfo(): Promise<JSON> {
        const fileInfo: FileInfo = await withLock(globals.paths.experimentsList, () => this.readExperimentsInfo());
        const experimentsInformation = JSON.parse(fileInfo.buffer.toString());
        const expIdList: Array<string> = Object.keys(experimentsInformation).filter((expId) => {
            return experimentsInformation[expId]['status'] !== 'STOPPED';
        });
        const updateList: Array<CrashedInfo> = (await Promise.all(expIdList.map((expId) => {
            return this.checkCrashed(expId, experimentsInformation[expId]['pid']);
        }))).filter(crashedInfo => crashedInfo.isCrashed);
        if (updateList.length > 0){
            const result = await withLock(globals.paths.experimentsList, () => {
                return this.updateAllStatus(updateList.map(crashedInfo => crashedInfo.experimentId), fileInfo.mtime)
            });
            if (result !== undefined) {
                return JSON.parse(JSON.stringify(Object.keys(result).map(key=>result[key])));
            } else {
                await timersPromises.setTimeout(500);
                return await this.getExperimentsInfo();
            }
        } else {
            return JSON.parse(JSON.stringify(Object.keys(experimentsInformation).map(key=>experimentsInformation[key])));
        }
    }

    public setExperimentInfo(experimentId: string, key: string, value: any): void {
        try {
            if (this.profileUpdateTimer[key] !== undefined) {
                // if a new call with the same timerId occurs, destroy the unfinished old one
                clearTimeout(this.profileUpdateTimer[key]!);
                this.profileUpdateTimer[key] = undefined;
            }
            withLockNoWait(globals.paths.experimentsList, () => {
                const experimentsInformation = JSON.parse(fs.readFileSync(globals.paths.experimentsList).toString());
                assert(experimentId in experimentsInformation, `Experiment Manager: Experiment Id ${experimentId} not found, this should not happen`);
                if (value !== undefined) {
                    experimentsInformation[experimentId][key] = value;
                } else {
                    delete experimentsInformation[experimentId][key];
                }
                fs.writeFileSync(globals.paths.experimentsList, JSON.stringify(experimentsInformation, null, 4));
            });
        } catch (err) {
            logger.error(err);
            logger.debug(`Experiment Manager: Retry set key value: ${experimentId} {${key}: ${value}}`);
            if ((err as any).code === 'EEXIST' || (err as any).message === 'File has been locked.') {
                this.profileUpdateTimer[key] = setTimeout(() => this.setExperimentInfo(experimentId, key, value), 100);
            }
        }
    }

    private readExperimentsInfo(): FileInfo {
        const buffer: Buffer = fs.readFileSync(globals.paths.experimentsList);
        const mtime: number = fs.statSync(globals.paths.experimentsList).mtimeMs;
        return {buffer: buffer, mtime: mtime};
    }

    private async checkCrashed(expId: string, pid: number): Promise<CrashedInfo> {
        const alive: boolean = await isAlive(pid);
        return {experimentId: expId, isCrashed: !alive}
    }

    private updateAllStatus(updateList: Array<string>, timestamp: number): {[key: string]: any} | undefined {
        if (timestamp !== fs.statSync(globals.paths.experimentsList).mtimeMs) {
            return;
        } else {
            const experimentsInformation = JSON.parse(fs.readFileSync(globals.paths.experimentsList).toString());
            updateList.forEach((expId: string) => {
                if (experimentsInformation[expId]) {
                    experimentsInformation[expId]['status'] = 'STOPPED';
                    delete experimentsInformation[expId]['port'];
                } else {
                    logger.error(`Experiment Manager: Experiment Id ${expId} not found, this should not happen`);
                }
            });
            fs.writeFileSync(globals.paths.experimentsList, JSON.stringify(experimentsInformation, null, 4));
            return experimentsInformation;
        }
    }

    private async cleanUp(): Promise<void> {
        const deferred = new Deferred<void>();
        if (this.isUndone()) {
            logger.debug('Experiment manager: something undone');
            setTimeout(((deferred: Deferred<void>): void => {
                if (this.isUndone()) {
                    deferred.reject(new Error('Still has undone after 5s, forced stop.'));
                } else {
                    deferred.resolve();
                }
            }).bind(this), 5 * 1000, deferred);
        } else {
            logger.debug('Experiment manager: all clean up');
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
