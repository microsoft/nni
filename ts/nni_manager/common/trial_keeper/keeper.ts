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

import { HttpChannelServer } from 'common/command_channel/http';
import globals from 'common/globals';
import { GpuScheduler } from 'common/gpu_scheduler';
import { Logger, getLogger } from 'common/log';
import { TrialProcess } from './process';

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
    private gpuScheduler: GpuScheduler | null = null;
    private log: Logger;
    private platform: string;
    private trials: Map<string, TrialProcess> = new Map();

    constructor(environmentId: string, platform: string, enableGpuScheduler: boolean) {
        this.envId = environmentId;
        this.platform = platform;
        this.log = getLogger(`TrialKeeper.${environmentId}`);

        this.channels = new HttpChannelServer(this.envId, `/env/${this.envId}`);

        if (enableGpuScheduler) {
            this.gpuScheduler = new GpuScheduler();
        }

        this.channels.onReceive((trialId, command) => {
            if (command.type === 'request_parameter') {
                this.emitter.emit('request-parameter', trialId);
            } else if (command.type === 'metric') {
                this.emitter.emit('metric', trialId, command.metric);
            } else {
                this.log.error('Received bad command:', trialId, command);
            }
        });
    }

    public async start(): Promise<void> {
        await this.channels.start();

        if (this.gpuScheduler !== null) {
            await this.gpuScheduler.init();
        }
    }

    public async shutdown(): Promise<void> {
        const trials = Array.from(this.trials.values());
        const promises = trials.map(trial => trial.kill());
        await Promise.all(promises);

        await this.channels.shutdown();
    }

    public registerDirectory(name: string, path: string): void {
        this.dirs.set(name, path);
    }

    public async createTrial(options: TrialKeeper.TrialOptions): Promise<string | null> {
        const trialId = options.id;

        let gpus: number[] | null = null;
        if (options.gpuNumber) {
            if (this.gpuScheduler === null) {
                this.log.error('GPU scheduler is not enabled');
                return null;
            }
            gpus = await this.gpuScheduler.schedule(
                globals.args.experimentId,
                trialId,
                options.gpuNumber,
                options.gpuRestrictions
            );
            if (gpus === null) {
                this.log.info('No GPU available');
                return null;
            }
        } else if (options.gpuNumber === 0) {
            gpus = [];  // hide all gpus
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
        });

        const procOptions = {
            command: options.command,
            codeDirectory: this.dirs.get(options.codeDirectoryName)!,
            outputDirectory: outputDir,
            commandChannelUrl: this.channels.getChannelUrl(trialId),
            gpuIndices: gpus ?? undefined,  // change null to undefined
            platform: this.platform,
            sequenceId: options.sequenceId,
        }

        const success = await trial.spawn(procOptions);
        if (success) {
            this.trials.set(trialId, trial);
            return trialId;
        } else {
            return null;
        }
    }

    public async stopTrial(trialId: string): Promise<void> {
        await this.trials.get(trialId)!.kill();
    }

    public async sendParameter(trialId: string, parameter: any): Promise<void> {
        const command = { type: 'parameter', parameter };
        this.channels.send(trialId, command);
    }

    public onTrialStart(callback: (trialId: string, timestamp: number) => void): void {
        this.emitter.on('trial-start', callback);
    }

    public onTrialStop(callback: (trialId: string, timestamp: number, exitCode: number | null) => void): void {
        this.emitter.on('trial-stop', callback);
    }

    public onRequestParameter(callback: (trialId: string) => void): void {
        this.emitter.on('request-parameter', callback);
    }

    public onMetric(callback: (trialId: string, metric: any) => void): void {
        this.emitter.on('metric', callback);
    }
}
