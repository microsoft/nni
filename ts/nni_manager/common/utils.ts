// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert';
import { randomBytes } from 'crypto';
import cpp from 'child-process-promise';
import cp from 'child_process';
import { ChildProcess, spawn, StdioOptions } from 'child_process';
import dgram from 'dgram';
import fs from 'fs';
import net from 'net';
import path from 'path';
import * as timersPromises from 'timers/promises';
import { Deferred } from 'ts-deferred';

import { Database, DataStore } from './datastore';
import globals from './globals';
import { resetGlobals } from './globals/unittest';  // TODO: this file should not contain unittest helpers
import { IocShim } from './ioc_shim';
import { ExperimentConfig, Manager } from './manager';
import { HyperParameters, TrainingService, TrialJobStatus } from './trainingService';

function getExperimentRootDir(): string {
    return globals.paths.experimentRoot;
}

function getLogDir(): string {
    return globals.paths.logDirectory;
}

function getLogLevel(): string {
    return globals.args.logLevel;
}

function getDefaultDatabaseDir(): string {
    return path.join(getExperimentRootDir(), 'db');
}

function getCheckpointDir(): string {
    return path.join(getExperimentRootDir(), 'checkpoint');
}

async function mkDirP(dirPath: string): Promise<void> {
    await fs.promises.mkdir(dirPath, { recursive: true });
}

function mkDirPSync(dirPath: string): void {
    fs.mkdirSync(dirPath, { recursive: true });
}

const delay = timersPromises.setTimeout;

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

/**
 * Generate command line to start automl algorithm(s),
 * either start advisor or start a process which runs tuner and assessor
 *
 * @param expParams: experiment startup parameters
 *
 */
function getMsgDispatcherCommand(expParams: ExperimentConfig): string[] {
    const clonedParams = Object.assign({}, expParams);
    delete clonedParams.searchSpace;
    return [ globals.args.pythonInterpreter, '-m', 'nni', '--exp_params', Buffer.from(JSON.stringify(clonedParams)).toString('base64') ];
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
    IocShim.snapshot(Database);
    IocShim.snapshot(DataStore);
    IocShim.snapshot(TrainingService);
    IocShim.snapshot(Manager);

    resetGlobals();

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
    IocShim.restore(Manager);
    IocShim.restore(TrainingService);
    IocShim.restore(DataStore);
    IocShim.restore(Database);
}

let cachedIpv4Address: string | null = null;

/**
 * Get IPv4 address of current machine.
 */
async function getIPV4Address(): Promise<string> {
    if (cachedIpv4Address !== null) {
        return cachedIpv4Address;
    }

    // creates "udp connection" to a non-exist target, and get local address of the connection.
    // since udp is connectionless, this does not send actual packets.
    const socket = dgram.createSocket('udp4');
    socket.connect(1, '192.0.2.0');
    for (let i = 0; i < 10; i++) {  // wait the system to initialize "connection"
        await timersPromises.setTimeout(1);
        try {
            cachedIpv4Address = socket.address().address;
            socket.close();
            return cachedIpv4Address;
        } catch (error) {
            /* retry */
        }
    }

    cachedIpv4Address = socket.address().address;  // if it still fails, throw the error
    socket.close();
    return cachedIpv4Address;
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
    const delayTimeout: Promise<number> = new Promise((_resolve: any, reject: (reason: Error) => any): void => {
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

/**
 * get the version of current package
 */
async function getVersion(): Promise<string> {
    const deferred: Deferred<string> = new Deferred<string>();
    import(path.join(__dirname, '..', 'package.json')).then((pkg) => {
        deferred.resolve(pkg.version);
    }).catch(() => {
        deferred.resolve('999.0.0-developing');
    });
    return deferred.promise;
}

/**
 * run command as ChildProcess
 */
function getTunerProc(command: string[], stdio: StdioOptions, newCwd: string, newEnv: any, newShell: boolean = true, isDetached: boolean = false): ChildProcess {
    // FIXME: TensorBoard has no reason to use get TUNER proc
    if (process.platform === "win32") {
        newShell = false;
        isDetached = true;
    }
    const tunerProc: ChildProcess = spawn(command[0], command.slice(1), {
        stdio,
        cwd: newCwd,
        env: newEnv,
        shell: newShell,
        detached: isDetached
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

async function isPortOpen(host: string, port: number): Promise<boolean> {
    return new Promise<boolean>((resolve, reject) => {
        try{
            const stream = net.createConnection(port, host);
            const id = setTimeout(() => {
                stream.destroy();
                resolve(false);
            }, 1000);

            stream.on('connect', () => {
                clearTimeout(id);
                stream.destroy();
                resolve(true);
            });

            stream.on('error', () => {
                clearTimeout(id);
                stream.destroy();
                resolve(false);
            });
        } catch (error) {
            reject(error);
        }
    });
}

async function getFreePort(host: string, start: number, end: number): Promise<number> {
    if (start > end) {
        throw new Error(`no more free port`);
    }
    if (await isPortOpen(host, start)) {
        return await getFreePort(host, start + 1, end);
    } else {
        return start;
    }
}

export function importModule(modulePath: string): any {
    module.paths.unshift(path.dirname(modulePath));
    return require(path.basename(modulePath));
}

export {
    countFilesRecursively, generateParamFileName, getMsgDispatcherCommand, getCheckpointDir,
    getLogDir, getExperimentRootDir, getJobCancelStatus, getDefaultDatabaseDir, getIPV4Address, unixPathJoin, getFreePort, isPortOpen,
    mkDirP, mkDirPSync, delay, prepareUnitTest, cleanupUnitTest, uniqueString, randomInt, randomSelect, getLogLevel, getVersion, getTunerProc, isAlive, killPid, getNewLine
};
