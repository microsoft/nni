/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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
