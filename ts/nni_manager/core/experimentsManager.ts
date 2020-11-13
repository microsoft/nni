// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as lockfile from 'lockfile';
import * as os from 'os';
import * as path from 'path';
import * as request from 'request';
import { Deferred } from 'ts-deferred';

import { getLogger, Logger } from '../common/log';
import { ExperimentStatus } from '../common/manager'
import { isAlive } from '../common/utils';


interface LockOpts {
    stale?: number;
    retries?: number;
    retryWait?: number;
}

interface AliveInfo {
    alive: boolean;
    timestamp: number;
}

interface NewStatusInfo {
    experimentId: string;
    newStatus: ExperimentStatus;
    timestamp: number;
}

class ExperimentsManager {
    private experimentsPath: string;
    private experimentsPathLock: string;
    private log: Logger;
    private lockOpts: LockOpts;

    constructor() {
        this.experimentsPath = path.join(os.homedir(), 'nni-experiments', '.experiment');
        this.experimentsPathLock = path.join(os.homedir(), 'nni-experiments', '.experiment.lock');
        this.log = getLogger();
        this.lockOpts = {stale: 2 * 1000, retries: 100, retryWait: 100};
    }

    public async getExperimentsInfo(): Promise<JSON> {
        const experimentsInformation = JSON.parse((await this.readExperimentsInfo()).toString());
        const expIdList: Array<string> = Object.keys(experimentsInformation).filter((expId) => {
            return experimentsInformation[expId]['status'] !== 'STOPPED';
        });
        const updateList: Array<NewStatusInfo> = await Promise.all(expIdList.map((expId) => {
            return this.getUpdatedStatus(expId, experimentsInformation[expId]['pid'], experimentsInformation[expId]['port']);
        }));
        return this.withLock(this.updateStatus, updateList, this.experimentsPath);
    }

    private async readExperimentsInfo(): Promise<Buffer> {
        return this.withLock(fs.readFileSync, this.experimentsPath);
    }

    private async withLock(func: Function, ...args: any): Promise<any> {
        const deferred = new Deferred<any>();
        lockfile.lock(this.experimentsPathLock, this.lockOpts, (err) => {
            if (err) {
                deferred.reject(err);
            }
            try {
                this.log.debug('Experiments Manager: .experiment locked');
                const result = func(...args);
                lockfile.unlockSync(this.experimentsPathLock);
                this.log.debug('Experiments Manager: .experiment unlocked');
                deferred.resolve(result);
            } catch (err) {
                deferred.reject(err);
            }
        });
        return deferred.promise;
    }

    private async isAlive(pid: number): Promise<AliveInfo> {
        const deferred: Deferred<AliveInfo> = new Deferred<AliveInfo>();
        isAlive(pid).then((alive: boolean) => {
            deferred.resolve({alive: alive, timestamp: Date.now()});
        }).catch((err) => {
            deferred.reject(err);
        });
        return deferred.promise;
    }

    private async getUpdatedStatus(expId: string, pid: number, port: number): Promise<NewStatusInfo> {
        const deferred: Deferred<NewStatusInfo> = new Deferred<NewStatusInfo>();
        const alive: AliveInfo = await this.isAlive(pid);
        if (alive.alive) {
            request(`http://localhost:${port}/api/v1/nni/check-status`, {json: true}, (err, res, body) => {
                if (err) {
                    deferred.resolve({experimentId: expId, newStatus: 'ERROR', timestamp: Date.now()})
                } else {
                    deferred.resolve({experimentId: expId, newStatus: body.status, timestamp: Date.now()})
                }
            });
        } else {
            deferred.resolve({experimentId: expId, newStatus: 'STOPPED', timestamp: alive.timestamp});
        }
        return deferred.promise;
    }

    private updateStatus(updateList: Array<NewStatusInfo>, experimentsPath: string): JSON {
        const experimentsInformation = JSON.parse(fs.readFileSync(experimentsPath).toString());
        updateList.forEach((newStatusInfo: NewStatusInfo) => {
            if (experimentsInformation[newStatusInfo.experimentId]) {
                if (experimentsInformation[newStatusInfo.experimentId]['statusUpdateTime'] < newStatusInfo.timestamp) {
                    experimentsInformation[newStatusInfo.experimentId]['status'] = newStatusInfo.newStatus;
                    experimentsInformation[newStatusInfo.experimentId]['statusUpdateTime'] = newStatusInfo.timestamp;
                }
            }
        });
        fs.writeFileSync(experimentsPath, JSON.stringify(experimentsInformation));
        return experimentsInformation;
    }
}

export { ExperimentsManager };
