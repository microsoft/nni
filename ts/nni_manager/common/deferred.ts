// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  TODO: Back ported from 3.0 draft.
 *
 *  An augmented version of ts-deferred.
 *
 *  You can `await deferred.promise` more than once and they will be resolved together.
 *
 *  You can resolve a deferred multiple times with identical value and it will be ignored.
 *
 *  If a deferred is resolved and/or rejected with conflict values,
 *  it will throw error and log both values or reasons.
 **/

import util from 'util';

import { Logger, getLogger } from 'common/log';

const logger = getLogger('common.deferred');

export class Deferred<T> {
    private resolveCallbacks: any[] = [];
    private rejectCallbacks: any[] = [];
    private isResolved: boolean = false;
    private isRejected: boolean = false;
    private resolvedValue?: T;
    private rejectedReason?: Error;

    public get promise(): Promise<T> {
        // use getter to compat ts-deferred
        if (this.isResolved) {
            return Promise.resolve(this.resolvedValue) as Promise<T>;
        }
        if (this.isRejected) {
            return Promise.reject(this.rejectedReason) as Promise<T>;
        }
        return new Promise<T>((resolutionFunc, rejectionFunc) => {
            this.resolveCallbacks.push(resolutionFunc);
            this.rejectCallbacks.push(rejectionFunc);
        });
    }

    public get settled(): boolean {
        // use getter for consistent api style
        return this.isResolved || this.isRejected;
    }

    public resolve = (value: T): void => {
        if (!this.isResolved && ! this.isRejected) {
            this.isResolved = true;
            this.resolvedValue = value;
            for (const callback of this.resolveCallbacks) {
                callback(value);
            }

        } else if (this.isResolved && this.resolvedValue == value) {
            logger.debug('Double resolve:', value);

        } else {
            const msg = this.errorMessage('trying to resolve with value: ' + util.inspect(value));
            logger.error(msg);
            throw new Error('Conflict Deferred result. ' + msg);
        }
    }

    public reject = (reason: Error): void => {
        if (!this.isResolved && !this.isRejected) {
            this.isRejected = true;
            this.rejectedReason = reason;
            for (const callback of this.rejectCallbacks) {
                callback(reason);
            }

        } else if (this.isRejected) {
            logger.warning('Double reject:', this.rejectedReason, reason);

        } else {
            const msg = this.errorMessage('trying to reject with reason: ' + util.inspect(reason));
            logger.error(msg);
            throw new Error('Conflict Deferred result. ' + msg);
        }
    }

    private errorMessage(curStat: string): string {
        let prevStat = '';
        if (this.isResolved) {
            prevStat = 'Already resolved with value: ' + util.inspect(this.resolvedValue);
        }
        if (this.isRejected) {
            prevStat = 'Already rejected with reason: ' + util.inspect(this.rejectedReason);
        }
        return prevStat + ' ; ' + curStat;
    }
}
