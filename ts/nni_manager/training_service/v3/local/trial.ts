// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import child_process, { ChildProcess, SpawnOptions } from 'child_process';
import events, { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';

import { Deferred } from 'common/deferred';
import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';

type TrialStatus = 'init' | 'running' | 'end' | 'error';

export class Trial {
    private emitter: EventEmitter;
    private id: string;
    private initResult: Deferred<boolean> = new Deferred();
    private log: Logger;
    private proc: ChildProcess | null = null;
    private status: TrialStatus = 'init';
    private stopped: Deferred<void> = new Deferred();

    constructor(id: string, emitter: EventEmitter) {
        this.id = id;
        this.emitter = emitter;
        this.log = getLogger(`Trial.${id}`);
    }

    public spawn(
            command: string,
            codeDirectory: string,
            outputDirectory: string,
            extraConfig: Record<string, string>): Promise<boolean> {
        this.spawnProc(command, codeDirectory, outputDirectory, extraConfig);
        return this.initResult.promise;
    }

    public isRunning(): boolean {
        return this.proc !== null && !this.stopped.settled
    }

    public kill(): Promise<void> {
        if (!this.isRunning()) {
            return Promise.resolve();
        }

        this.log.debug('Kill trial');

        if (process.platform === 'win32') {
            this.proc!.kill();  // TODO: how to graceful kill?
        } else {
            this.proc!.kill('SIGTERM');
            setTimeout(() => {
                if (this.isRunning()) {
                    this.log.info('Failed to terminate in 5s. Send SIGKILL');
                    this.proc!.kill('SIGKILL');
                }
            }, 5000);
        }
        return this.stopped.promise;
    }

    private emitStart(): void {
        this.log.info('Trial start');
        this.emitter.emit('trial_start', this.id, Date.now());
    }

    private emitStop(code: number | null, signal: string | null): void {
        this.log.info(`Trial end: exitCode=${code} signal=${signal}`);
        this.emitter.emit('trial_end', this.id, Date.now(), code);
    }

    private async spawnProc(
            command: string,
            codeDir: string,
            outputDir: string,
            extra: Record<string, string>): Promise<void> {
        // TODO: move these to a config file for better debuggability
        const env = Object.assign({}, process.env, extra);
        env['NNI_CODE_DIR'] = codeDir;
        env['NNI_EXP_ID'] = globals.args.experimentId;
        env['NNI_OUTPUT_DIR'] = outputDir;
        env['NNI_SYS_DIR'] = outputDir;
        env['NNI_TRIAL_JOB_ID'] = this.id;
        // in the new architecture training services are plain, isolated training platforms
        // they should not aware experiment-wise status like the global trail counter
        // instead, there could be a parameter like `extraTrialEnv` from nni manager
        env['NNI_TRIAL_SEQ_ID'] = String(-1);

        const stdout = fs.createWriteStream(path.join(outputDir, 'trial.stdout'));
        const stderr = fs.createWriteStream(path.join(outputDir, 'trial.stderr'));
        await Promise.all([
            events.once(stdout, 'open'),
            events.once(stderr, 'open'),
        ]);

        const options: SpawnOptions = {  // cannot resolve overload without type hint (tsc bug)
            cwd: codeDir,
            env,
            stdio: [ 'ignore', stdout, stderr ],
            shell: (process.platform === 'win32' ? true : '/bin/bash'),  // use bash rather than sh
        }

        this.proc = child_process.spawn(command, options);
        this.proc.on('close', this.handleClose.bind(this));
        this.proc.on('error', this.handleError.bind(this));
        this.proc.on('exit', this.handleExit.bind(this));
        this.proc.on('spawn', this.handleSpawn.bind(this));
    }

    private handleSpawn(): void {
        if (this.status === 'init') {
            this.status = 'running';
            this.initResult.resolve(true);
            this.emitStart();
            return;
        }

        this.log.warning(`Received spawn event in ${this.status} status`);
    }

    private handleExit(code: number | null, signal: string | null): void {
        if (this.status === 'init') {
            this.log.warning('Trial exited immediately');
            this.status = 'end';
            this.initResult.resolve(true);
            this.stopped.resolve();
            this.emitStart();
            this.emitStop(code, signal);
            return;
        }

        if (this.status === 'running') {
            this.status = 'end';
            this.stopped.resolve();
            this.emitStop(code, signal);
            return
        }

        this.log.warning(`Received close event in ${this.status} status`);
    }

    private handleError(err: Error): void {
        this.log.error(`Trial error in ${this.status} status:`, err);

        if (this.status === 'init') {
            this.status = 'error';
            this.initResult.resolve(false);
            setTimeout(() => { this.stopped.resolve(); }, 1000);
            return;
        }

        if (this.status === 'running') {
            this.status = 'error';
            setTimeout(() => {
                if (!this.stopped.settled) {
                    this.stopped.resolve();
                    this.emitStop(null, null);
                }
            }, 1000);
            return;
        }

        this.status = 'error';
    }

    private handleClose(code: number | null, signal: string | null): void {
        if (!this.stopped.settled) {
            this.stopped.resolve();
            this.emitStop(code, signal);
        }
        this.proc = null;
        this.log.debug('Trial cleaned');
    }
}
