// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

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
import { ExperimentStartupInfo, getExperimentStartupInfo, setExperimentStartupInfo } from './experimentStartupInfo';
import { ExperimentParams, Manager } from './manager';
import { HyperParameters, TrainingService, TrialJobStatus } from './trainingService';
import { logLevelNameMap } from './log';

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

function randomInt(max: number): number {
    return Math.floor(Math.random() * max);
}

function randomSelect<T>(a: T[]): T {
    assert(a !== undefined);

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

function getCmdPy(): string {
    let cmd = 'python3';
    if (process.platform === 'win32') {
        cmd = 'python';
    }
    return cmd;
}

/**
 * Generate command line to start automl algorithm(s),
 * either start advisor or start a process which runs tuner and assessor
 *
 * @param expParams: experiment startup parameters
 *
 */
function getMsgDispatcherCommand(expParams: ExperimentParams): string {
    const clonedParams = Object.assign({}, expParams);
    delete clonedParams.searchSpace;
    return `${getCmdPy()} -m nni --exp_params ${Buffer.from(JSON.stringify(clonedParams)).toString('base64')}`;
}

/**
 * Generate parameter file name based on HyperParameters object
 * @param hyperParameters HyperParameters instance
 */
function generateParamFileName(hyperParameters: HyperParameters): string {
    assert(hyperParameters !== undefined);
    assert(hyperParameters.index >= 0);

    let paramFileName: string;
    if (hyperParameters.index == 0) {
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

    const logLevel: string = parseArg(['--log_level', '-ll']);
    if (logLevel.length > 0 && !logLevelNameMap.has(logLevel)) {
        console.log(`FATAL: invalid log_level: ${logLevel}`);
    }

    setExperimentStartupInfo(true, 'unittest', 8080, 'unittest', undefined, logLevel);
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

let cachedipv4Address: string = '';
/**
 * Get IPv4 address of current machine
 */
function getIPV4Address(): string {
    if (cachedipv4Address && cachedipv4Address.length > 0) {
        return cachedipv4Address;
    }

    const networkInterfaces = os.networkInterfaces();
    if (networkInterfaces.eth0) {
        for (const item of networkInterfaces.eth0) {
            if (item.family === 'IPv4') {
                cachedipv4Address = item.address;
                return cachedipv4Address;
            }
        }
    } else {
        throw Error(`getIPV4Address() failed because os.networkInterfaces().eth0 is undefined. Please specify NNI manager IP in config.`);
    }

    throw Error('getIPV4Address() failed because no valid IPv4 address found.')
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
function countFilesRecursively(directory: string): Promise<number> {
    if (!fs.existsSync(directory)) {
        throw Error(`Direcotory ${directory} doesn't exist`);
    }

    const deferred: Deferred<number> = new Deferred<number>();

    let timeoutId: NodeJS.Timer
    const delayTimeout: Promise<number> = new Promise((resolve: Function, reject: Function): void => {
        // Set timeout and reject the promise once reach timeout (5 seconds)
        timeoutId = setTimeout(() => {
            reject(new Error(`Timeout: path ${directory} has too many files`));
        }, 5000);
    });

    let fileCount: number = -1;
    let cmd: string;
    if (process.platform === "win32") {
        cmd = `powershell "Get-ChildItem -Path ${directory} -Recurse -File | Measure-Object | %{$_.Count}"`
    } else {
        cmd = `find ${directory} -type f | wc -l`;
    }
    cpp.exec(cmd).then((result) => {
        if (result.stdout && parseInt(result.stdout)) {
            fileCount = parseInt(result.stdout);
        }
        deferred.resolve(fileCount);
    });
    return Promise.race([deferred.promise, delayTimeout]).finally(() => {
        clearTimeout(timeoutId);
    });
}

export function validateFileName(fileName: string): boolean {
    const pattern: string = '^[a-z0-9A-Z._-]+$';
    const validateResult = fileName.match(pattern);
    if (validateResult) {
        return true;
    }
    return false;
}

async function validateFileNameRecursively(directory: string): Promise<boolean> {
    if (!fs.existsSync(directory)) {
        throw Error(`Direcotory ${directory} doesn't exist`);
    }

    const fileNameArray: string[] = fs.readdirSync(directory);
    let result = true;
    for (const name of fileNameArray) {
        const fullFilePath: string = path.join(directory, name);
        try {
            // validate file names and directory names
            result = validateFileName(name);
            if (fs.lstatSync(fullFilePath).isDirectory()) {
                result = result && await validateFileNameRecursively(fullFilePath);
            }
            if (!result) {
                return Promise.reject(new Error(`file name in ${fullFilePath} is not valid!`));
            }
        } catch (error) {
            return Promise.reject(error);
        }
    }
    return Promise.resolve(result);
}

/**
 * get the version of current package
 */
async function getVersion(): Promise<string> {
    const deferred: Deferred<string> = new Deferred<string>();
    import(path.join(__dirname, '..', 'package.json')).then((pkg) => {
        deferred.resolve(pkg.version);
    }).catch((error) => {
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
    if (process.platform === "win32") {
        cmd = command.split(" ", 1)[0];
        arg = command.substr(cmd.length + 1).split(" ");
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
async function isAlive(pid: any): Promise<boolean> {
    const deferred: Deferred<boolean> = new Deferred<boolean>();
    let alive: boolean = false;
    if (process.platform === 'win32') {
        try {
            const str = cp.execSync(`powershell.exe Get-Process -Id ${pid} -ErrorAction SilentlyContinue`).toString();
            if (str) {
                alive = true;
            }
        }
        catch (error) {
            //ignore
        }
    }
    else {
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
async function killPid(pid: any): Promise<void> {
    const deferred: Deferred<void> = new Deferred<void>();
    try {
        if (process.platform === "win32") {
            await cpp.exec(`cmd.exe /c taskkill /PID ${pid} /F`);
        }
        else {
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
    else {
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

export {
    countFilesRecursively, validateFileNameRecursively, generateParamFileName, getMsgDispatcherCommand, getCheckpointDir,
    getLogDir, getExperimentRootDir, getJobCancelStatus, getDefaultDatabaseDir, getIPV4Address, unixPathJoin,
    mkDirP, mkDirPSync, delay, prepareUnitTest, parseArg, cleanupUnitTest, uniqueString, randomInt, randomSelect, getLogLevel, getVersion, getCmdPy, getTunerProc, isAlive, killPid, getNewLine
};
