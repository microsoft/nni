/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import * as fs from 'fs';
import * as request from 'request';
import * as path from 'path';
import { Deferred } from 'ts-deferred';
import * as component from "../../common/component";
import { delay, generateParamFileName, getExperimentRootDir } from "../../common/utils";
import { KILL_TRIAL_JOB, NEW_TRIAL_JOB } from '../../core/commands';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { encodeCommand } from "../../core/ipcInterface";
import { EnvironmentInformation } from "./environment";
import { TrialDetail, TrialService } from "./trial";
import { PythonShell } from 'python-shell';
import { TrialJobApplicationForm } from "../../common/trainingService";
import { AMLClusterConfig, AMLTrialConfig, AMLTrialJobDetail } from '../aml/amlConfig';
import { AMLTrainingService } from 'training_service/aml/amlTrainingService';
import { AMLEnvironmentService } from './amlEnvironmentService';

@component.Singleton
export class AMLTrialService extends TrialService {

    private amlClusterConfig: AMLClusterConfig | undefined;
    private amlTrialConfig: AMLTrialConfig | undefined;
    private experimentId: string;
    private amlEnvironmentService: AMLEnvironmentService;

    constructor() {
        super();
        this.amlEnvironmentService = component.get(AMLEnvironmentService);
        this.amlClusterConfig = this.amlEnvironmentService.amlClusterConfig;
        this.amlTrialConfig = this.amlEnvironmentService.amlTrialConfig;
        this.experimentId = getExperimentId();
    }

    public async config(_key: string, _value: string): Promise<void> {
        return;
    }

    public async refreshTrialsStatus(trials: TrialDetail[]): Promise<void> {
        for (const trial of trials) {
            const currentStatus = trial.status;
            // to prevent inconsistent status, skip all non running trials
            if (currentStatus !== "RUNNING") {
                continue;
            }

            const environment = trial.environment;
            if (environment === undefined) {
                this.log.error(`found running trial ${trial.id} has no environment, set trial to UNKNOWN.`);
                trial.status = "UNKNOWN";
                continue;
            }

            console.log('--------update trial status-------')
        }
    }

    public async startTrial(trial: TrialDetail): Promise<void> {
        console.log('-----------79 start trial--------')
        if (trial.environment === undefined) {
            throw new Error(`trialService: environment of trial ${trial.id} shouldn't be undefined!`);
        }
        await this.sendCommand(NEW_TRIAL_JOB, trial.settings, trial.environment);
    }

    public async stopTrial(trial: TrialDetail): Promise<void> {
        if (trial.environment === undefined) {
            throw new Error(`trialService: environment of trial ${trial.id} shouldn't be undefined!`);
        }
        await this.sendCommand(KILL_TRIAL_JOB, trial.id, trial.environment);
    }

    public async updateTrial(trial: TrialDetail, form: TrialJobApplicationForm): Promise<void> {
    }

    private async sendCommand(commantType: string, data: any, environment: EnvironmentInformation): Promise<void> {
        console.log('---------------96 send command0----------')
        const deferred: Deferred<void> = new Deferred<void>();
        const command = encodeCommand(commantType, JSON.stringify(data));
        let fileName = `manager_command_${new Date().getTime()}.txt`;
        let filePath = path.join(environment.environmentLocalTempFolder, fileName);
        await fs.promises.writeFile(filePath, command.toString("utf8"), { encoding: 'utf8' });
        
        if (this.amlClusterConfig === undefined) {
            throw new Error('AML Cluster config is not initialized');
        }
        if (this.amlTrialConfig === undefined) {
            throw new Error('AML trial config is not initialized');
        }

        let pyshell = new PythonShell('uploadFile.py', {
            scriptPath: './config/aml',
            pythonOptions: ['-u'], // get print results in real-time
            args: [
                '--subscription_id', this.amlClusterConfig.subscriptionId,
                '--resource_group', this.amlClusterConfig.resourceGroup,
                '--workspace_name', this.amlClusterConfig.workspaceName,
                '--experiment_name', `nni_exp_${this.experimentId}`,
                '--environment_id', environment.id,
                '--remote_file_name', fileName,
                '--local_file_path', filePath,
              ]
        });
        pyshell.on('message', function (result: any) {
            // received a message sent from the Python script (a simple "print" statement)
            console.log(`============upload data======${result}`);
            deferred.resolve();
        });
        return deferred.promise;
    }
}
