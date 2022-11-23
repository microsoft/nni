// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  The helper class to implement a "reuse" training service.
 *
 *  TrialKeeper has a very similar interface to TrainingSerivceV3.
 *  In fact the local training service is a thin wrapper over TrialKeeper.
 **/

import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';

import type { Command } from 'common/command_channel/interface';
import { HttpChannelServer } from 'common/command_channel/http';
import globals from 'common/globals';
import { Logger, getLogger } from 'common/log';
import { TrialProcess, TrialProcessOptions } from './process';
import { GpuSchedulerClient } from './gpu_scheduler_client';

export declare namespace TrialKeeper {
    export interface TrialOptions {
        id: string;
        command: string;
        codeDirectoryName: string;
        sequenceId?: number;
        gpuNumber?: number;
        gpuRestrictions?: GpuRestrictions;
    }

    export interface GpuRestrictions {
        onlyUseIndices?: number[];
        rejectActive?: boolean;
        rejectComputeActive?: boolean;
    }
}

export class TrialKeeper {
    private envId: string;
    private channels: HttpChannelServer;
    private dirs: Map<string, string> = new Map();
    private emitter: EventEmitter = new EventEmitter();
    private gpuScheduler: GpuSchedulerClient;
    private log: Logger;
    private platform: string;
    private trials: Map<string, TrialProcess> = new Map();

    constructor(environmentId: string, platform: string, enableGpuScheduler: boolean) {
        this.envId = environmentId;
        this.platform = platform;
        this.log = getLogger(`TrialKeeper.${environmentId}`);

        this.gpuScheduler = new GpuSchedulerClient(enableGpuScheduler);

        this.channels = new HttpChannelServer(this.envId, `/env/${this.envId}`);
        this.channels.onReceive((trialId, command) => {
            this.emitter.emit('command', trialId, command);
            if (command.type !== 'request_parameter' && command.type !== 'metric') {
                this.log.warning(`Unexpected command from trial ${trialId}:`, command);
            }
        });
    }

    // TODO: support user configurable init command
    public async start(): Promise<void> {
        await Promise.all([
            this.gpuScheduler.start(),
            this.channels.start(),
        ]);
    }

    public async shutdown(): Promise<void> {
        let promises: Promise<void>[] = [
            this.gpuScheduler.shutdown(),
            this.channels.shutdown(),
        ];

        const trials = Array.from(this.trials.values());
        promises = promises.concat(trials.map(trial => trial.kill()));

        await Promise.all(promises);
    }

    public registerDirectory(name: string, path: string): void {
        this.dirs.set(name, path);
    }

    // FIXME: the method name will be changed when we support distributed trials
    public async createTrial(options: TrialKeeper.TrialOptions): Promise<boolean> {
        const trialId = options.id;

        const gpuEnv = await this.gpuScheduler.schedule(trialId, options.gpuNumber, options.gpuRestrictions);
        if (gpuEnv === null) {
            this.log.info('No GPU available');
            return false;
        }

        // TODO: move this to globals.paths
        const outputDir = path.join(globals.paths.experimentRoot, 'environments', this.envId, 'trials', trialId);
        await fs.mkdir(outputDir, { recursive: true });

        const trial = new TrialProcess(trialId);
        trial.onStart(timestamp => {
            this.emitter.emit('trial-start', trialId, timestamp);
        });
        trial.onStop((timestamp, exitCode, _signal) => {
            this.emitter.emit('trial-stop', trialId, timestamp, exitCode);
            this.gpuScheduler.release(trialId);  // TODO: fire and forget, handle exception?
        });

        const procOptions: TrialProcessOptions = {
            command: options.command,
            codeDirectory: this.dirs.get(options.codeDirectoryName)!,
            outputDirectory: outputDir,
            commandChannelUrl: this.channels.getChannelUrl(trialId),
            platform: this.platform,
            sequenceId: options.sequenceId,
            environmentVariables: gpuEnv,
        }

        const success = await trial.spawn(procOptions);
        if (success) {
            this.trials.set(trialId, trial);
            return true;
        } else {
            return false;
        }
    }

    public async stopTrial(trialId: string): Promise<void> {
        await this.trials.get(trialId)!.kill();
    }

    public async sendCommand(trialId: string, command: Command): Promise<void> {
        this.channels.send(trialId, command);
    }

    public onTrialStart(callback: (trialId: string, timestamp: number) => void): void {
        this.emitter.on('trial-start', callback);
    }

    public onTrialStop(callback: (trialId: string, timestamp: number, exitCode: number | null) => void): void {
        this.emitter.on('trial-stop', callback);
    }

    public onReceiveCommand(commandType: string, callback: (trialId: string, command: Command) => void): void {
        this.emitter.on('command', (trialId, command) => {
            if (command.type === commandType) {
                callback(trialId, command);
            }
        });
    }
}
