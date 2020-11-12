// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as lockfile from 'lockfile';
import * as os from 'os';
import * as path from 'path';
import * as request from 'request';
import { update } from 'tar';
import { Deferred } from 'ts-deferred';

import { getLogger, Logger } from '../common/log';
import { isAlive } from '../common/utils';


interface LockOpts {
    stale?: number;
    retries?: number;
    retryWait?: number;
}

interface experimentsInfo {
    [key: string]: any
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
        const experimentsInformation: experimentsInfo = JSON.parse((await this.readExperimentsInfo()).toString());
        let updateList: {[key: string]: any} = {};
        for (let expId in experimentsInformation) {
            if (experimentsInformation[expId]['status'] !== 'STOPPED'){
                let newStatus: string = 'STOPPED';
                let updateTime: number = Date.now();
                if (isAlive(experimentsInformation[expId]['pid'])) {
                    request(`http://localhost:${experimentsInformation[expId]['port']}/api/v1/nni/check-status`, {json: true}, (err, res, body) => {
                        if (err) {
                            newStatus = 'ERROR';
                            updateTime = Date.now();
                        }
                        newStatus = body.status;
                        updateTime = Date.now();
                    });
                }
                updateList[expId] = {'status': newStatus, 'statusUpdateTime': updateTime};
            }
        }
    }

    private async readExperimentsInfo(): Promise<Buffer> {
        return this.withLock(fs.readFileSync, this.experimentsPath);
    }

    private async withLock (func: Function, ...args: any): Promise<any> {
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
}

export { ExperimentsManager };
