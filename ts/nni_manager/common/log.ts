// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import { Writable } from 'stream';
import { WritableStreamBuffer } from 'stream-buffers';
import { format } from 'util';
import * as component from '../common/component';
import { getExperimentStartupInfo, isReadonly } from './experimentStartupInfo';

export const DEBUG = 10;
export const INFO = 20;
export const WARNING = 30;
export const ERROR = 40;
export const CRITICAL = 50;

export const TRACE = 1;
export const FATAL = 50;

const levelNames: Record<number, string> = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    TRACE: 'TRACE',
}

let logFile: Writable | null = null;

export function setLevel(level: number | string) {

}

export function start(logFileName: string) {
    logFile = fs.createWriteStream(logFile, {
        flags: 'a+',
        encoding: 'utf8',
        autoClose: true
    });
}

export function stop() {
    logFile.end();
    logFile = null;
}

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
        if (level < logLevel || logFile === null) {
            return;
        }

        // time.toLocaleString('sv') trick does not work for Windows
        const time1 = new Date(new Date().toLocaleString() + ' UTC').toISOString();
        const time = time1.slice(0, 10) + ' ' + time1.slice(11, 19);

        const levelName: string = levelNames[level] === undefined ? level.toString() : levelNames[level];

        const words = [];
        for (const arg of args) {
            if (arg === undefined) {
                words.push('undefined');
            } else if (arg === null) {
                words.push('null');
            } else if (typeof arg === 'object') {
                const json = JSON.stringify(arg);
                words.push(json === undefined ? arg : json);
            } else {
                words.push(arg);
            }
        }
        const message = words.join(' ');
        
        const record = `[${time}] ${levelName} (${this.name}) ${message}\n`;
        logFile.write(record);
    }
}

const loggers = new Map<string, Logger>();

export function getLogger(name: string = 'root'): Logger {
    let logger = loggers.get(name);
    if (logger === undefined) {
        logger = new Logger(name);
        loggers.set(name, logger);
    }
    return logger;
}
