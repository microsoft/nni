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

import { randomBytes } from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Deferred } from 'ts-deferred';
import { Container } from 'typescript-ioc';
import * as util from 'util';

import { Database, DataStore } from './datastore';
import { ExperimentStartupInfo, getExperimentId, setExperimentStartupInfo } from './experimentStartupInfo';
import { Manager } from './manager';
import { TrainingService } from './trainingService';

function getExperimentRootDir(): string{
    return path.join(os.homedir(), 'nni', 'experiments', getExperimentId());
}

function getLogDir(): string{
    return path.join(getExperimentRootDir(), 'log');
}

function getDefaultDatabaseDir(): string {
    return path.join(getExperimentRootDir(), 'db');
}

function mkDirP(dirPath: string): Promise<void> {
    const deferred: Deferred<void> = new Deferred<void>();
    fs.exists(dirPath, (exists: boolean) => {
        if (exists) {
            deferred.resolve();
        } else {
            const parent: string = path.dirname(dirPath);
            mkDirP(parent).then(() => {
                fs.mkdir(dirPath, (err: Error) => {
                    if (err) {
                        deferred.reject(err);
                    } else {
                        deferred.resolve();
                    }
                });
            }).catch((err: Error) => {
                deferred.reject(err);
            });
        }
    });

    return deferred.promise;
}

function mkDirPSync(dirPath: string): void {
    if (fs.existsSync(dirPath)) {
        return;
    }
    mkDirPSync(path.dirname(dirPath));
    fs.mkdirSync(dirPath);
}

const delay: (ms: number) => Promise<void> = util.promisify(setTimeout);

/**
 * Convert index to character
 * @param index index
 * @returns a mapping character
 */
function charMap(index: number): number {
    if (index < 26) {
        return index + 97;
    } else if (index < 52) {
        return index - 26 + 65;
    } else {
        return index - 52 + 48;
    }
}

/**
 * Generate a unique string by length
 * @param len length of string
 * @returns a unique string
 */
function uniqueString(len: number): string {
    if (len === 0) {
        return '';
    }
    const byteLength: number = Math.ceil((Math.log2(52) + Math.log2(62) * (len - 1)) / 8);
    let num: number = randomBytes(byteLength).reduce((a: number, b: number) => a * 256 + b, 0);
    const codes: number[] = [];
    codes.push(charMap(num % 52));
    num = Math.floor(num / 52);
    for (let i: number = 1; i < len; i++) {
        codes.push(charMap(num % 62));
        num = Math.floor(num / 62);
    }

    return String.fromCharCode(...codes);
}

function parseArg(names: string[]): string {
    if (process.argv.length >= 4) {
        for (let i: number = 2; i < process.argv.length - 1; i++) {
            if (names.includes(process.argv[i])) {
                return process.argv[i + 1];
            }
        }
    }

    return '';
}

/**
 * Generate command line to start advisor process which runs tuner and assessor
 * @param tuner : For builtin tuner:
 *     {
 *         className: 'EvolutionTuner'
 *         classArgs: {
 *             optimize_mode: 'maximize',
 *             population_size: 3
 *         }
 *     }
 * customized:
 *     {
 *         codeDir: '/tmp/mytuner'
 *         classFile: 'best_tuner.py'
 *         className: 'BestTuner'
 *         classArgs: {
 *             optimize_mode: 'maximize',
 *             population_size: 3
 *         }
 *     }
 *
 * @param assessor: similiar as tuner
 *
 */
function getMsgDispatcherCommand(tuner: any, assessor: any): string {
    let command: string = `python3 -m nni --tuner_class_name ${tuner.className}`;

    if (tuner.classArgs !== undefined) {
        command += ` --tuner_args ${JSON.stringify(JSON.stringify(tuner.classArgs))}`;
    }

    if (tuner.codeDir !== undefined && tuner.codeDir.length > 1) {
        command += ` --tuner_directory ${tuner.codeDir}`;
    }

    if (tuner.classFileName !== undefined && tuner.classFileName.length > 1) {
        command += ` --tuner_class_filename ${tuner.classFileName}`;
    }

    if (assessor !== undefined && assessor.className !== undefined) {
        command += ` --assessor_class_name ${assessor.className}`;
        if (assessor.classArgs !== undefined) {
            command += ` --assessor_args ${JSON.stringify(JSON.stringify(assessor.classArgs))}`;
        }

        if (assessor.codeDir !== undefined && assessor.codeDir.length > 1) {
            command += ` --assessor_directory ${assessor.codeDir}`;
        }

        if (assessor.classFileName !== undefined && assessor.classFileName.length > 1) {
            command += ` --assessor_class_filename ${assessor.classFileName}`;
        }
    }

    return command;
}

/**
 * Initialize a pseudo experiment environment for unit test.
 * Must be paired with `cleanupUnitTest()`.
 */
function prepareUnitTest(): void {
    Container.snapshot(ExperimentStartupInfo);
    Container.snapshot(Database);
    Container.snapshot(DataStore);
    Container.snapshot(TrainingService);
    Container.snapshot(Manager);

    setExperimentStartupInfo(true, 'unittest');
    mkDirPSync(getLogDir());

    const sqliteFile: string = path.join(getDefaultDatabaseDir(), 'nni.sqlite');
    try {
        fs.unlinkSync(sqliteFile);
    } catch (err) {
        // file not exists, good
    }
}

/**
 * Clean up unit test pseudo experiment.
 * Must be paired with `prepareUnitTest()`.
 */
function cleanupUnitTest(): void {
    Container.restore(Manager);
    Container.restore(TrainingService);
    Container.restore(DataStore);
    Container.restore(Database);
    Container.restore(ExperimentStartupInfo);
}

export { getMsgDispatcherCommand, getLogDir, getExperimentRootDir, getDefaultDatabaseDir, mkDirP, delay, prepareUnitTest,
    parseArg, cleanupUnitTest, uniqueString };
