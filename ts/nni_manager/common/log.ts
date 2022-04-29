// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Python-like logging interface.
 *
 *      const logger = getLogger('moduleName');
 *      logger.info('hello', { to: 'world' });
 *
 *  Outputs:
 *
 *      [1970-01-01 00:00:00] INFO (moduleName) hello { to: 'world' }
 *
 *  Loggers use `util.inspect()` to format values,
 *  so objects will be smartly stringified and exceptions will include stack trace.
 **/

import util from 'util';

import globals from 'common/globals';

const levelNameToValue = { trace: 0, debug: 10, info: 20, warning: 30, error: 40, critical: 50 } as const;

const loggers: Record<string, Logger> = {};

export function getLogger(name: string): Logger {
    if (loggers[name] === undefined) {
        loggers[name] = new Logger(name);
    }
    return loggers[name];
}

/**
 *  A special logger prints to stderr when the logging system has problems.
 *  For modules that are responsible for handling logger errors.
 **/
export function getRobustLogger(name: string): Logger {
    if (loggers[name] === undefined || !(loggers[name] as RobustLogger).robust) {
        loggers[name] = new RobustLogger(name);
    }
    return loggers[name];
}

export class Logger {
    protected name: string;

    constructor(name: string) {
        this.name = name;
    }

    public trace(...args: any[]): void {
        this.log(levelNameToValue.trace, 'TRACE', args);
    }

    public debug(...args: any[]): void {
        this.log(levelNameToValue.debug, 'DEBUG', args);
    }

    public info(...args: any[]): void {
        this.log(levelNameToValue.info, 'INFO', args);
    }

    public warning(...args: any[]): void {
        this.log(levelNameToValue.warning, 'WARNING', args);
    }

    public error(...args: any[]): void {
        this.log(levelNameToValue.error, 'ERROR', args);
    }

    public critical(...args: any[]): void {
        this.log(levelNameToValue.critical, 'CRITICAL', args);
    }

    protected log(levelValue: number, levelName: string, args: any[]): void {
        if (levelValue >= levelNameToValue[globals.args.logLevel]) {
            const msg = `[${timestamp()}] ${levelName} (${this.name}) ${formatArgs(args)}`;
            globals.logStream.writeLine(msg);
        }
    }
}

class RobustLogger extends Logger {
    public readonly robust: boolean = true;
    private errorOccurred: boolean = false;

    protected log(levelValue: number, levelName: string, args: any[]): void {
        if (this.errorOccurred) {
            this.logAfterError(levelName, args);
            return;
        }
        try {
            if (levelValue >= levelNameToValue[globals.args.logLevel]) {
                const msg = `[${timestamp()}] ${levelName} (${this.name}) ${formatArgs(args)}`;
                globals.logStream.writeLineSync(msg);
            }
        } catch (error) {
            this.errorOccurred = true;
            console.error('[ERROR] Logger has stopped working:', error);
            this.logAfterError(levelName, args);
        }
    }

    private logAfterError(levelName: string, args: any[]): void {
        try {
            args = args.map(arg => util.inspect(arg));
        } catch { /* fallback */ }
        console.error(`[${levelName}] (${this.name})`, ...args);
    }
}

function timestamp(): string {
    const now = new Date();
    const date = now.getFullYear() + '-' + zeroPad(now.getMonth() + 1) + '-' + zeroPad(now.getDate());
    const time = zeroPad(now.getHours()) + ':' + zeroPad(now.getMinutes()) + ':' + zeroPad(now.getSeconds());
    return date + ' ' + time;
}

function zeroPad(num: number): string {
    return num.toString().padStart(2, '0');
}

function formatArgs(args: any[]): string {
    return args.map(arg => (typeof arg === 'string' ? arg : util.inspect(arg))).join(' ');
}
