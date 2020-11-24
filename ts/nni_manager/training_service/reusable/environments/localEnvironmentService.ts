// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import * as request from 'request';
import { Deferred } from 'ts-deferred';
import * as tkill from 'tree-kill';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { PAIClusterConfig } from '../../pai/paiConfig';
import { NNIPAIK8STrialConfig } from '../../pai/paiK8S/paiK8SConfig';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { StorageService } from '../storageService';
import { TrialConfig } from '../../common/trialConfig';
import { getExperimentRootDir, isAlive } from '../../../common/utils';
import { LocalConfig } from '../../local/localTrainingService';
import { execMkdir, validateCodeDir, runScript, fileExist, execCopydir } from '../../common/util';


@component.Singleton
export class LocalEnvironmentService extends EnvironmentService {

    private readonly log: Logger = getLogger();
    private localTrialConfig: TrialConfig | undefined;
    private localConfig: LocalConfig | undefined;
    private experimentRootDir: string;
    private experimentId: string;

    constructor() {
        super();
        this.experimentId = getExperimentId();
        this.experimentRootDir = getExperimentRootDir();
    }

    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public async config(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.LOCAL_CONFIG:
                this.localConfig = <LocalConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                this.localTrialConfig = <TrialConfig>JSON.parse(value);
                break;
            default:
                this.log.debug(`OpenPAI not proccessed metadata key: '${key}', value: '${value}'`);
        }
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        const tasks: Promise<void>[] = [];
        environments.forEach(async (environment) => {
            tasks.push(this.refreshEnvironment(environment));
        });
        await Promise.all(tasks);
    }

    public async refreshEnvironment(environment: EnvironmentInformation): Promise<void> {
        const jobpidPath: string = `${environment.runnerWorkingFolder}/pid`;
        const runnerReturnCodeFilePath: string = `${environment.runnerWorkingFolder}/code`;
        /* eslint-disable require-atomic-updates */
        try {
            // check if pid file exist
            const pidExist = await fileExist(jobpidPath);
            if (!pidExist) {
                return;
            }
            const pid: string = await fs.promises.readFile(jobpidPath, 'utf8');
            const alive: boolean = await isAlive(pid);
            environment.status = 'RUNNING';
            // if the process of jobpid is not alive any more
            if (!alive) {
                if (fs.existsSync(runnerReturnCodeFilePath)) {
                    const runnerReturnCode: string = await fs.promises.readFile(runnerReturnCodeFilePath, 'utf8');
                    const match: RegExpMatchArray | null = runnerReturnCode.trim()
                        .match(/^-?(\d+)\s+(\d+)$/);
                    if (match !== null) {
                        const { 1: code } = match;
                        // Update trial job's status based on result code
                        if (parseInt(code, 10) === 0) {
                            environment.setStatus('SUCCEEDED');
                        } else {
                            environment.setStatus('FAILED');
                        }
                    }
                }
            }
        } catch (error) {
            this.log.error(`Update job status exception, error is ${error.message}`);
        }
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.localTrialConfig === undefined) {
            throw new Error('Local trial config is not initialized');
        }
        if (this.localConfig === undefined) {
            throw new Error('Local config is not initialized');
        }
        const localEnvironment: LocalEnvironmentInformation = environment as LocalEnvironmentInformation;
        // Need refactor, this temp folder path is not appropriate, there are two expId in this path
        const localTempFolder: string = path.join(this.experimentRootDir, this.experimentId,
            "environment-temp", "envs");
        const localEnvCodeFolder: string = path.join(this.experimentRootDir, "envs");
        localEnvironment.runnerWorkingFolder = path.join(localEnvCodeFolder, localEnvironment.id);
        await execMkdir(localEnvironment.runnerWorkingFolder);
        await execCopydir(localTempFolder, localEnvCodeFolder);
        localEnvironment.command = `cd ${this.experimentRootDir} && \
${localEnvironment.command} --job_pid_file ${localEnvironment.runnerWorkingFolder}/pid \
1>${localEnvironment.runnerWorkingFolder}/trialrunner_stdout 2>${localEnvironment.runnerWorkingFolder}/trialrunner_stderr \
&& echo $? \`date +%s%3N\` >${localEnvironment.runnerWorkingFolder}/code`;
        await fs.promises.writeFile(path.join(localEnvCodeFolder, 'nni_run.sh'),
        localEnvironment.command, { encoding: 'utf8', mode: 0o777 }),
        // Execute command in local machine
        runScript(path.join(localEnvCodeFolder, 'nni_run.sh'));
        localEnvironment.trackingUrl = `${localEnvironment.runnerWorkingFolder}`;
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        const jobpidPath: string = `${environment.runnerWorkingFolder}/pid`;
        const pid: string = await fs.promises.readFile(jobpidPath, 'utf8');
        tkill(Number(pid), 'SIGKILL');
    }
}
