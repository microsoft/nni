// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import { Writable } from 'stream';
import * as util from 'util';

/* log level constants */

export const DEBUG = 10;
export const INFO = 20;
export const WARNING = 30;
export const ERROR = 40;
export const CRITICAL = 50;

export const TRACE = 1;
export const FATAL = 50;

const levelNames = new Map<number, string>([
    [CRITICAL, 'CRITICAL'],
    [ERROR, 'ERROR'],
    [WARNING, 'WARNING'],
    [INFO, 'INFO'],
    [DEBUG, 'DEBUG'],
    [TRACE, 'TRACE'],
]);

/* global_ states */

let logLevel: number = 0;
const loggers = new Map<string, Logger>();

/* major api */

export class Logger {
    private name: string;

    constructor(name: string = 'root') {
        this.name = name;
    }

    public trace(...args: any[]): void {
        this.log(TRACE, args);
    }

    public debug(...args: any[]): void {
        this.log(DEBUG, args);
    }

    public info(...args: any[]): void {
        this.log(INFO, args);
    }

    public warning(...args: any[]): void {
        this.log(WARNING, args);
    }

    public error(...args: any[]): void {
        this.log(ERROR, args);
    }

    public critical(...args: any[]): void {
        this.log(CRITICAL, args);
    }

    public fatal(...args: any[]): void {
        this.log(FATAL, args);
    }

    private log(level: number, args: any[]): void {
        const logFile: Writable | undefined = (global as any).logFile;
        if (level < logLevel) {
            return;
        }

        // `time.toLocaleString('sv')` trick does not work for Windows
        const isoTime = new Date(new Date().toLocaleString() + ' UTC').toISOString();
        const time = isoTime.slice(0, 10) + ' ' + isoTime.slice(11, 19);

        const levelName = levelNames.has(level) ? levelNames.get(level) : level.toString();

        const message = args.map(arg => (typeof arg === 'string' ? arg : util.inspect(arg))).join(' ');
        
        const record = `[${time}] ${levelName} (${this.name}) ${message}\n`;

        if (logFile === undefined) {
            console.log(record);
        } else {
            logFile.write(record);
        }
    }
}

export function getLogger(name: string = 'root'): Logger {
    let logger = loggers.get(name);
    if (logger === undefined) {
        logger = new Logger(name);
        loggers.set(name, logger);
    }
    return logger;
}

/* management functions */

export function setLogLevel(levelName: string): void {
    if (levelName) {
        const level = module.exports[levelName.toUpperCase()];
        if (typeof level === 'number') {
            logLevel = level;
        } else {
            console.log('[ERROR] Bad log level:', levelName);
            getLogger('logging').error('Bad log level:', levelName);
        }
    }
}

export function startLogging(logPath: string): void {
    (global as any).logFile = fs.createWriteStream(logPath, {
        flags: 'a+',
        encoding: 'utf8',
        autoClose: true
    });
}

export function stopLogging(): void {
    if ((global as any).logFile !== undefined) {
        (global as any).logFile.end();
        (global as any).logFile = undefined;
    }
}
