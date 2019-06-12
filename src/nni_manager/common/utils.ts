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

import * as assert from 'assert';
import { randomBytes } from 'crypto';
import * as cpp from 'child-process-promise';
import * as cp from 'child_process';
import { ChildProcess, spawn, StdioOptions } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Deferred } from 'ts-deferred';
import { Container } from 'typescript-ioc';
import * as util from 'util';

import { Database, DataStore } from './datastore';
import { ExperimentStartupInfo, getExperimentId, getExperimentStartupInfo, setExperimentStartupInfo } from './experimentStartupInfo';
import { Manager } from './manager';
import { TrialConfig } from '../training_service/common/trialConfig';
import { HyperParameters, TrainingService, TrialJobStatus } from './trainingService';
import { getLogger } from './log';

function getExperimentRootDir(): string {
    return getExperimentStartupInfo()
            .getLogDir();
}

function getLogDir(): string {
    return path.join(getExperimentRootDir(), 'log');
}

function getLogLevel(): string {
    return getExperimentStartupInfo()
    .getLogLevel();
}

function getDefaultDatabaseDir(): string {
    return path.join(getExperimentRootDir(), 'db');
}

function getCheckpointDir(): string {
    return path.join(getExperimentRootDir(), 'checkpoint');
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

function randomSelect<T>(a: T[]): T {
    assert(a !== undefined);

    // tslint:disable-next-line:insecure-random
    return a[Math.floor(Math.random() * a.length)];
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

function encodeCmdLineArgs(args: any): any {
    if(process.platform === 'win32'){
        return JSON.stringify(args);
    }
    else{
        return JSON.stringify(JSON.stringify(args));
    }
}

function getCmdPy(): string {
    let cmd = 'python3';
    if(process.platform === 'win32'){
        cmd = 'python';
    }
    return cmd;
}

/**
 * Generate command line to start automl algorithm(s), 
 * either start advisor or start a process which runs tuner and assessor
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
 * @param advisor: similar as tuner
 *
 */
function getMsgDispatcherCommand(tuner: any, assessor: any, advisor: any, multiPhase: boolean = false, multiThread: boolean = false): string {
    if ((tuner || assessor) && advisor) {
        throw new Error('Error: specify both tuner/assessor and advisor is not allowed');
    }
    if (!tuner && !advisor) {
        throw new Error('Error: specify neither tuner nor advisor is not allowed');
    }
    let command: string = `${getCmdPy()} -m nni`;
    if (multiPhase) {
        command += ' --multi_phase';
    }

    if (multiThread) {
        command += ' --multi_thread';
    }

    if (advisor) {
        command += ` --advisor_class_name ${advisor.className}`;
        if (advisor.classArgs !== undefined) {
            command += ` --advisor_args ${encodeCmdLineArgs(advisor.classArgs)}`;
        }
        if (advisor.codeDir !== undefined && advisor.codeDir.length > 1) {
            command += ` --advisor_directory ${advisor.codeDir}`;
        }
        if (advisor.classFileName !== undefined && advisor.classFileName.length > 1) {
            command += ` --advisor_class_filename ${advisor.classFileName}`;
        }
    } else {
        command += ` --tuner_class_name ${tuner.className}`;
        if (tuner.classArgs !== undefined) {
            command += ` --tuner_args ${encodeCmdLineArgs(tuner.classArgs)}`;
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
                command += ` --assessor_args ${encodeCmdLineArgs(assessor.classArgs)}`;
            }
            if (assessor.codeDir !== undefined && assessor.codeDir.length > 1) {
                command += ` --assessor_directory ${assessor.codeDir}`;
            }
            if (assessor.classFileName !== undefined && assessor.classFileName.length > 1) {
                command += ` --assessor_class_filename ${assessor.classFileName}`;
            }
        }
    }

    return command;
}

/**
 * Generate parameter file name based on HyperParameters object
 * @param hyperParameters HyperParameters instance
 */
function generateParamFileName(hyperParameters : HyperParameters): string {
    assert(hyperParameters !== undefined);
    assert(hyperParameters.index >= 0);

    let paramFileName : string;
    if(hyperParameters.index == 0) {
        paramFileName = 'parameter.cfg';
    } else {
        paramFileName = `parameter_${hyperParameters.index}.cfg`
    }
    return paramFileName;
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

    setExperimentStartupInfo(true, 'unittest', 8080);
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

let cachedipv4Address : string = '';
/**
 * Get IPv4 address of current machine
 */
function getIPV4Address(): string {
    if (cachedipv4Address && cachedipv4Address.length > 0) {
        return cachedipv4Address;
    }

    if(os.networkInterfaces().eth0) {
        for(const item of os.networkInterfaces().eth0) {
            if(item.family === 'IPv4') {
                cachedipv4Address = item.address;
                return cachedipv4Address;
            }
        }
    } else {
        throw Error('getIPV4Address() failed because os.networkInterfaces().eth0 is undefined.');
    }

    throw Error('getIPV4Address() failed because no valid IPv4 address found.')
}

function getRemoteTmpDir(osType: string): string {
    if (osType == 'linux') {
        return '/tmp';
    } else {
        throw Error(`remote OS ${osType} not supported`);
    }
}

/**
 * Get the status of canceled jobs according to the hint isEarlyStopped
 */
function getJobCancelStatus(isEarlyStopped: boolean): TrialJobStatus {
    return isEarlyStopped ? 'EARLY_STOPPED' : 'USER_CANCELED';
}

/**
 * Utility method to calculate file numbers under a directory, recursively
 * @param directory directory name
 */
function countFilesRecursively(directory: string, timeoutMilliSeconds?: number): Promise<number> {
    if(!fs.existsSync(directory)) {
        throw Error(`Direcotory ${directory} doesn't exist`);
    }

    const deferred: Deferred<number> = new Deferred<number>();

    let timeoutId : NodeJS.Timer
    const delayTimeout : Promise<number> = new Promise((resolve : Function, reject : Function) : void => {
        // Set timeout and reject the promise once reach timeout (5 seconds)
        timeoutId = setTimeout(() => {
            reject(new Error(`Timeout: path ${directory} has too many files`));
        }, 5000);
    });

    let fileCount: number = -1;
    let cmd: string;
    if(process.platform === "win32") {
        cmd = `powershell "Get-ChildItem -Path ${directory} -Recurse -File | Measure-Object | %{$_.Count}"`
    } else {
        cmd = `find ${directory} -type f | wc -l`;   
    }
    cpp.exec(cmd).then((result) => {
        if(result.stdout && parseInt(result.stdout)) {
            fileCount = parseInt(result.stdout);            
        }
        deferred.resolve(fileCount);
    });
    return Promise.race([deferred.promise, delayTimeout]).finally(() => {
        clearTimeout(timeoutId);
    });
}

/**
 * get the version of current package
 */
async function getVersion(): Promise<string> {
    const deferred : Deferred<string> = new Deferred<string>();
    import(path.join(__dirname, '..', 'package.json')).then((pkg)=>{
        deferred.resolve(pkg.version);
    }).catch((error)=>{
        deferred.reject(error);
    });
    return deferred.promise;
} 

/**
 * run command as ChildProcess
 */
function getTunerProc(command: string, stdio: StdioOptions, newCwd: string, newEnv: any): ChildProcess {
    let cmd: string = command;
    let arg: string[] = [];
    let newShell: boolean = true;
    if(process.platform === "win32"){
        cmd = command.split(" ", 1)[0];
        arg = command.substr(cmd.length+1).split(" ");
        newShell = false;
    }
    const tunerProc: ChildProcess = spawn(cmd, arg, {
        stdio,
        cwd: newCwd,
        env: newEnv,
        shell: newShell
    });
    return tunerProc;
}

/**
 * judge whether the process is alive
 */
async function isAlive(pid:any): Promise<boolean> {
    let deferred : Deferred<boolean> = new Deferred<boolean>();
    let alive: boolean = false;
    if(process.platform ==='win32'){
        try {
            const str = cp.execSync(`powershell.exe Get-Process -Id ${pid} -ErrorAction SilentlyContinue`).toString();
            if (str) {
                alive = true;
            }
        }
        catch (error) {
        }
    }
    else{
        try {
            await cpp.exec(`kill -0 ${pid}`);
            alive = true;
        } catch (error) {
            //ignore
        }
    }
    deferred.resolve(alive);
    return deferred.promise;
}

/**
 * kill process 
 */
async function killPid(pid:any): Promise<void> {
    let deferred : Deferred<void> = new Deferred<void>();
    try {
        if (process.platform === "win32") {
            await cpp.exec(`cmd /c taskkill /PID ${pid} /F`);
        }
        else{
            await cpp.exec(`kill -9 ${pid}`);
        }
    } catch (error) {
        // pid does not exist, do nothing here
    }
    deferred.resolve();
    return deferred.promise;
}

function getNewLine(): string {
    if (process.platform === "win32") {
        return "\r\n";
    }
    else{
        return "\n";
    }
}

/**
 * Use '/' to join path instead of '\' for all kinds of platform
 * @param path 
 */
function unixPathJoin(...paths: any[]): string {
    const dir: string = paths.filter((path: any) => path !== '').join('/');
    if (dir === '') return '.';
    return dir;
}

export {countFilesRecursively, getRemoteTmpDir, generateParamFileName, getMsgDispatcherCommand, getCheckpointDir,
    getLogDir, getExperimentRootDir, getJobCancelStatus, getDefaultDatabaseDir, getIPV4Address, unixPathJoin,
    mkDirP, delay, prepareUnitTest, parseArg, cleanupUnitTest, uniqueString, randomSelect, getLogLevel, getVersion, getCmdPy, getTunerProc, isAlive, killPid, getNewLine };
