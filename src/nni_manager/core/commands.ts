/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.

 * MIT License

 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


const INITIALIZE = 'IN';
const REQUEST_TRIAL_JOBS = 'GE';
const REPORT_METRIC_DATA = 'ME';
const UPDATE_SEARCH_SPACE = 'SS';
const ADD_CUSTOMIZED_TRIAL_JOB = 'AD';
const TRIAL_END = 'EN';
const TERMINATE = 'TE';
const PING = 'PI';

const INITIALIZED = 'ID';
const NEW_TRIAL_JOB = 'TR';
const SEND_TRIAL_JOB_PARAMETER = 'SP';
const NO_MORE_TRIAL_JOBS = 'NO';
const KILL_TRIAL_JOB = 'KI';

const TUNER_COMMANDS: Set<string> = new Set([
    INITIALIZE,
    REQUEST_TRIAL_JOBS,
    REPORT_METRIC_DATA,
    UPDATE_SEARCH_SPACE,
    ADD_CUSTOMIZED_TRIAL_JOB,
    TERMINATE,
    PING,

    INITIALIZED,
    NEW_TRIAL_JOB,
    SEND_TRIAL_JOB_PARAMETER,
    NO_MORE_TRIAL_JOBS
]);

const ASSESSOR_COMMANDS: Set<string> = new Set([
    INITIALIZE,
    REPORT_METRIC_DATA,
    TRIAL_END,
    TERMINATE,

    KILL_TRIAL_JOB
]);

export {
    INITIALIZE,
    REQUEST_TRIAL_JOBS,
    REPORT_METRIC_DATA,
    UPDATE_SEARCH_SPACE,
    ADD_CUSTOMIZED_TRIAL_JOB,
    TRIAL_END,
    TERMINATE,
    PING,
    INITIALIZED,
    NEW_TRIAL_JOB,
    NO_MORE_TRIAL_JOBS,
    KILL_TRIAL_JOB,
    TUNER_COMMANDS,
    ASSESSOR_COMMANDS,
    SEND_TRIAL_JOB_PARAMETER
};
