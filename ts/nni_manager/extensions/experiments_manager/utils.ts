// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'fs';
import path from 'path';
import * as timersPromises from 'timers/promises';

import glob from 'glob';
import lockfile from 'lockfile';

const lockStale: number = 2000;
const retry: number = 100;

export function withLockNoWait<T>(protectedFile: string, func: () => T): T {
    const lockName = path.join(path.dirname(protectedFile), path.basename(protectedFile) + `.lock.${process.pid}`);
    const lockPath = path.join(path.dirname(protectedFile), path.basename(protectedFile) + '.lock.*');
    const lockFileNames: string[] = glob.sync(lockPath);
    const canLock: boolean = lockFileNames.map((fileName) => {
        return fs.existsSync(fileName) && Date.now() - fs.statSync(fileName).mtimeMs < lockStale;
    }).filter(unexpired=>unexpired === true).length === 0;
    if (!canLock) {
        throw new Error('File has been locked.');
    }
    lockfile.lockSync(lockName, { stale: lockStale });
    const result = func();
    lockfile.unlockSync(lockName);
    return result;
}

export async function withLock<T>(protectedFile: string, func: () => T): Promise<T> {
    for (let i = 0; i < retry; i += 1) {
        try {
            return withLockNoWait(protectedFile, func);
        } catch (error: any) {
            if (error.code === 'EEXIST' || error.message === 'File has been locked.') {
                await timersPromises.setTimeout(50);
            } else {
                throw error;
            }
        }
    }
    throw new Error('Lock file out of retries.');
}
