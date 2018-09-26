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

const joi = require('joi');

export namespace ValidationSchemas {
    export const SETCLUSTERMETADATA = {
        body: {
            machine_list: joi.array().items(joi.object({
                username: joi.string().required(),
                ip: joi.string().ip().required(),
                port: joi.number().min(1).max(65535).required(),
                passwd: joi.string().required(),
                sshKeyPath: joi.string(),
                passphrase: joi.string()
            })),
            trial_config: joi.object({
                gpuNum: joi.number().min(0).required(),
                codeDir: joi.string().min(1).required(),
                command: joi.string().min(1).required()
            })
        }
    };
    export const STARTEXPERIMENT = {
        body: {
            experimentName: joi.string().required(),
            authorName: joi.string(),
            maxTrialNum: joi.number().min(0).required(),
            trialConcurrency: joi.number().min(0).required(),
            searchSpace: joi.string().required(),
            maxExecDuration: joi.number().min(0).required(),
            tuner: joi.object({
                builtinTunerName: joi.string().valid('TPE', 'Random', 'Anneal', 'Evolution', 'SMAC'),
                codeDir: joi.string(),
                classFileName: joi.string(),
                className: joi.string(),
                classArgs: joi.any(),
                gpuNum: joi.number().min(0),
                checkpointDir: joi.string()
            }).required(),
            assessor: joi.object({
                builtinAssessorName: joi.string().valid('Medianstop'),
                codeDir: joi.string(),
                classFileName: joi.string(),
                className: joi.string(),
                classArgs: joi.any(),
                gpuNum: joi.number().min(0),
                checkpointDir: joi.string()
            }),
            clusterMetaData: joi.array().items(joi.object({
                key: joi.string(),
                value: joi.any()
            }))
        }
    };
    export const UPDATEEXPERIMENT = {
        query: {
            update_type: joi.string().required().valid('TRIAL_CONCURRENCY', 'MAX_EXEC_DURATION', 'SEARCH_SPACE')
        },
        body: {
            id: joi.string().required(),
            revision: joi.number().min(0).required(),
            params: joi.object(STARTEXPERIMENT.body).required(),
            execDuration: joi.number().required(),
            startTime: joi.number(),
            endTime: joi.number()
        }
    };
    export const STARTTENSORBOARD = {
        query: {
            job_ids: joi.string().min(5).max(5).required()
        }
    };
    export const STOPTENSORBOARD = {
        query: {
            endpoint: joi.string().uri().required()
        }
    };
}
