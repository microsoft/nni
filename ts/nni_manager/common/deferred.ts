// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import util from 'util';

/**
 *  A more powerful `Deferred` that allows to await multiple times.
 *
 *  Example usage:
 *
 *      const deferred = new Deferred<void>();
 *      deferred.promise.then(() => { console.log('hello'); });
 *      deferred.promise.then(() => { console.log('world'); });
 *      deferred.resolve();
 *
 *  In the example above, both "hello" and "world" will be logged, in arbitrary order.
 **/
export class Deferred<T> {
    private resolveCallbacks: any[] = [];
    private rejectCallbacks: any[] = [];

    private isResolved: boolean = false;
    private resolvedValue!: T;

    private isRejected: boolean = false;
    private rejectedReason!: Error;

    public get promise(): Promise<T> {
        if (this.isResolved) {
            return Promise.resolve(this.resolvedValue);
        }
        if (this.isRejected) {
            return Promise.reject(this.rejectedReason);
        }

        const p = new Promise<T>((resolve, reject) => {
            this.resolveCallbacks.push(resolve);
            this.rejectCallbacks.push(reject);
        });
        return p;
    }

    public resolve(value: T): void {
        this.checkError('Trying to resolve:', value);

        this.isResolved = true;
        this.resolvedValue = value;

        for (const callback of this.resolveCallbacks) {
            callback(value);
        }
    }

    public reject(reason: Error): void {
        this.checkError('Trying to reject:', reason);

        this.isRejected = true;
        this.rejectedReason = reason;

        for (const callback of this.rejectCallbacks) {
            callback(reason);
        }
    }

    private checkError(message: string, value: any): void {
        if (this.isResolved) {
            const prev = util.inspect(this.resolvedValue);
            const cur = util.inspect(value);
            throw new Error(`Deferred has already been resolved: ${prev} ; ${message} ${cur}`);
        }
        if (this.isRejected) {
            const prev = util.inspect(this.rejectedReason);
            const cur = util.inspect(value);
            throw new Error(`Deferred has already been rejected: ${prev} ; ${message} ${cur}`);
        }
    }
}
