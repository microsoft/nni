// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Manage the lifecycle of a trial process.
 *
 *  This is trial keeper's internal helper.
 *  Should not be used elsewhere.
 **/

import child_process, { ChildProcess, SpawnOptions } from 'child_process';
import events from 'events';
import fs from 'fs';
import path from 'path';

import { Deferred } from 'common/deferred';
import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';

export interface TrialProcessOptions {
    command: string;
    shell?: string;
    codeDirectory: string;
    outputDirectory: string;
    commandChannelUrl: string;
    platform: string;
    sequenceId?: number;
    environmentVariables: Record<string, string>;
}

interface TrialProcessInfo {
    startSuccess?: boolean;  // it is true when the shell successfully launched, even if the trial command is ill-formed
    startTime?: number;
    stopTime?: number;
    stopCode?: number | null;
    stopSignal?: string | null;
}

export class TrialProcess {
    private id: string;
    private info: TrialProcessInfo = {};
    private log: Logger;
    private proc: ChildProcess | null = null;
    private started: Deferred<void> = new Deferred();
    private stopped: Deferred<void> = new Deferred();

    constructor(id: string) {
        this.id = id;
        this.log = getLogger(`TrialProcess.${id}`);
    }

    /**
     *  Spawn the trial process. Return success or not.
     *
     *  Note that the trial is considered "success" here even if the trial command is ill-formed
     *  or it returns non-zero code immediately.
     *
     *  This is because the command is run in a shell and
     *  this method only checks if the shell is successfully launched.
     *
     *  If `options.shell` is empty or not set, it will use "/bin/bash" on UNIX,
     *  or "ComSpec" environment variable on Windows (default behavior of Node.js).
     *  When the shell is powershell, the exit code is likely to be 0/1 instead of concrete number.
     **/
    public async spawn(options: TrialProcessOptions): Promise<boolean> {
        // might change for log collection
        const stdout = fs.createWriteStream(path.join(options.outputDirectory, 'trial.stdout'));
        const stderr = fs.createWriteStream(path.join(options.outputDirectory, 'trial.stderr'));
        await Promise.all([ events.once(stdout, 'open'), events.once(stderr, 'open') ]);

        let shell: string | boolean = true;
        if (options.shell) {
            shell = options.shell;
        } else if (process.platform !== 'win32') {
            shell = '/bin/bash';
        }

        const spawnOptions: SpawnOptions = {
            cwd: options.codeDirectory,
            env: this.buildEnv(options),
            stdio: [ 'ignore', stdout, stderr ],
            shell: shell,
        };

        this.proc = child_process.spawn(options.command, spawnOptions);

        this.proc.on('spawn', () => { this.resolveStart('spawn'); });
        this.proc.on('exit', (code, signal) => { this.resolveStop('exit', code, signal); });
        this.proc.on('error', (err) => { this.handleError(err); });
        this.proc.on('close', (code, signal) => { this.resolveStop('close', code, signal); });

        await this.started.promise;
        return Boolean(this.info.startSuccess);
    }

    /**
     *  Kill the trial, or silently do nothing if it is not running.
     *
     *  On Unix, send SIGINT signal and wait for `timeout` milliseconds; then force kill with SIGKILL.
     *  It uses SIGINT instead of SIGTERM because SIGINT can be caught as `KeyboardInterrupt` in python.
     *
     *  (FIXME) On Windows, always do force kill.
     *
     *  (FIXME) This only kills the trial process. If the trial has child processes, they are not touched by NNI.
     *
     *  The returned promise is resolved together with onStop() callback.
     **/
    public async kill(timeout?: number): Promise<void> {
        this.log.trace('kill');

        if (!this.started.settled) {
            this.log.error('Killing a not started trial');
            return;
        }
        if (this.stopped.settled) {
            return;
        }

        if (process.platform === 'win32') {
            this.proc!.kill();  // FIXME

        } else {
            this.proc!.kill('SIGINT');
            setTimeout(() => {
                if (!this.stopped.settled) {
                    this.log.info(`Failed to terminate in ${timeout ?? 5000} ms, force kill`);
                    this.proc!.kill('SIGKILL');
                }
            }, timeout ?? 5000);
        }

        await this.stopped.promise;
    }

    /**
     *  Register an on trial start callback.
     *
     *  The callback will always get invoked when spawn() returns true,
     *  even if this method is called after start complete.
     **/
    public onStart(callback: (timestamp: number) => void): void {
        this.started.promise.then(() => {
            if (this.info.startSuccess) {
                callback(this.info.startTime!);
            }
        });
    }

    /**
     *  Register an on trial stop callback.
     *
     *  The callback is guaranteed to be invoked as long as spawn() returns true,
     *  even if this method is called after trial stopped.
     *
     *  If the trial stopped proactively (including exited during a SIGINT handler),
     *  exitCode will be a number; otherwise it will be null.
     *
     *  Note: When the shell is powershell, the exit code is likely to be 0/1 instead of concrete number.
     **/
    public onStop(callback: (timestamp: number, exitCode: number | null, signal: string | null) => void): void {
        this.stopped.promise.then(() => {
            if (this.info.startSuccess) {
                callback(this.info.stopTime!, this.info.stopCode!, this.info.stopSignal!);
            }
        });
    }

    private buildEnv(opts: TrialProcessOptions): Record<string, string> {
        // TODO: use a config file instead of environment varaibles for better debuggability
        const env: Record<string, string> = { ...(process.env as any), ...opts.environmentVariables };
        env['NNI_CODE_DIR'] = opts.codeDirectory;
        env['NNI_EXP_ID'] = globals.args.experimentId;
        env['NNI_OUTPUT_DIR'] = opts.outputDirectory;
        env['NNI_PLATFORM'] = opts.platform;
        env['NNI_SYS_DIR'] = opts.outputDirectory;
        env['NNI_TRIAL_COMMAND_CHANNEL'] = opts.commandChannelUrl;
        env['NNI_TRIAL_JOB_ID'] = this.id;
        env['NNI_TRIAL_SEQ_ID'] = String(opts.sequenceId ?? -1);
        this.log.trace('Env:', env);
        return env;
    }

    private resolveStart(event: string): void {
        this.log.trace('Start', event);

        if (this.started.settled) {
            this.log.warning(`Receive ${event} event after started`);
            return;
        }

        this.info.startTime = Date.now();

        if (this.stopped.settled) {
            this.log.error(`Receive ${event} event after stopped`);
            this.info.startSuccess = false;
        } else {
            this.info.startSuccess = true;
        }

        this.started.resolve();
    }

    private resolveStop(event: string, code: number | null, signal: string | null, timestamp?: number): void {
        this.log.trace('Stop', event, code, signal, timestamp);

        if (event === 'close') {
            this.log.debug('Stopped and cleaned');
            this.proc = null;
        }

        if (this.stopped.settled) {
            if (event !== 'close' && timestamp === undefined) {
                this.log.warning(`Receive ${event} event after stopped`);
            }
            return;
        }

        this.info.stopTime = timestamp ?? Date.now();
        this.info.stopCode = code;
        this.info.stopSignal = signal;

        if (!this.started.settled) {
            this.log.error(`Receive ${event} event before starting`);
            this.info.startSuccess = false;
            this.started.resolve();
        }

        this.stopped.resolve();
    }

    private handleError(err: Error): void {
        this.log.error('Error:', err);
        if (!this.stopped.settled) {
            const time = Date.now();
            setTimeout(() => { this.resolveStop('error', null, null, time); }, 1000);
        }
    }
}
