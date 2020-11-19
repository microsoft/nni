// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import { getLogger, Logger } from '../common/log';
import { isAlive, withLock } from '../common/utils';
import { ExpManager } from '../common/expmanager';


interface LockOpts {
    stale?: number;
    retries?: number;
    retryWait?: number;
}

interface CrashedInfo {
    experimentId: string;
    isCrashed: boolean;
}

interface FileInfo {
    buffer: Buffer;
    mtime: number;
}

class ExperimentsManager implements ExpManager {
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
        const fileInfo = await this.readExperimentsInfo();
        const experimentsInformation = JSON.parse(fileInfo.buffer.toString());
        const expIdList: Array<string> = Object.keys(experimentsInformation).filter((expId) => {
            return experimentsInformation[expId]['status'] !== 'STOPPED';
        });
        const updateList: Array<CrashedInfo> = (await Promise.all(expIdList.map((expId) => {
            return this.checkCrashed(expId, experimentsInformation[expId]['pid']);
        }))).filter(crashedInfo => crashedInfo.isCrashed);
        if (updateList.length > 0){
            const result = await this.withLock(this.updateStatus, updateList.map(crashedInfo => crashedInfo.experimentId), fileInfo.mtime);
            if (result !== undefined) {
                return result;
            } else {
                return this.getExperimentsInfo();
            }
        } else {
            return experimentsInformation;
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
        this.experimentsPathLock = newPath + '.lock';
    }

    private async withLock (func: Function, ...args: any): Promise<any> {
        return withLock(func, this.experimentsPathLock, this.lockOpts, ...args);
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

    private updateStatus(updateList: Array<string>, timestamp: number) {
        if (timestamp !== fs.statSync(this.experimentsPath).mtimeMs) {
            return;
        } else {
            const experimentsInformation = JSON.parse(fs.readFileSync(this.experimentsPath).toString());
            updateList.forEach((expId: string) => {
                if (experimentsInformation[expId]) {
                    experimentsInformation[expId]['status'] = 'STOPPED';
                } else {
                    this.log.error('Experiment Manager: Experiment Id not found, this should not happen');
                }
            });
            fs.writeFileSync(this.experimentsPath, JSON.stringify(experimentsInformation));
            return experimentsInformation;
        }
    }
}

export { ExperimentsManager };
