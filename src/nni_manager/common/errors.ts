// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

export namespace NNIErrorNames {
    export const NOT_FOUND: string = 'NOT_FOUND';
    export const INVALID_JOB_DETAIL: string = 'NO_VALID_JOB_DETAIL_FOUND';
    export const RESOURCE_NOT_AVAILABLE: string = 'RESOURCE_NOT_AVAILABLE';
}

export class NNIError extends Error {
    public cause!: Error | undefined;
    constructor (name: string, message: string, err?: Error) {
        super(message);
        this.name = name;
        if (err !== undefined) {
            this.stack = err.stack;
        }
        this.cause = err;
    }

    public static FromError(err: NNIError | Error | string, messagePrefix?: string): NNIError {
        const msgPrefix: string = messagePrefix === undefined ? '' : messagePrefix;
        if (err instanceof NNIError) {
            if (err.message !== undefined) {
                err.message = msgPrefix + err.message;
            }

            return err;
        } else if (typeof(err) === 'string') {
            return new NNIError('', msgPrefix + err);
        } else if (err instanceof Error) {
            return new NNIError('', msgPrefix + err.message, err);
        } else {
            throw new Error(`Wrong instance type: ${typeof(err)}`);
        }
    }
}

export class MethodNotImplementedError extends Error {
    constructor() {
        super('Method not implemented.');
    }
}
