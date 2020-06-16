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

import * as component from "../../common/component";
import { delay, generateParamFileName } from "../../common/utils";
import { KILL_TRIAL_JOB, NEW_TRIAL_JOB } from '../../core/commands';
import { encodeCommand } from "../../core/ipcInterface";
import { EnvironmentInformation } from "./environment";
import { StorageService } from "./storageService";
import { TrialDetail, TrialService } from "./trial";
import { TrialJobApplicationForm } from "../../common/trainingService";

@component.Singleton
export class StorageTrialService extends TrialService {
    public async config(_key: string, _value: string): Promise<void> {
        return;
    }

    public async refreshTrialsStatus(trials: TrialDetail[]): Promise<void> {
        const storageService = component.get<StorageService>(StorageService);

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

            let remoteFiles: string[] = [];
            const codeFilePath = storageService.joinPath(trial.workingDirectory, trial.TRIAL_METADATA_DIR);
            remoteFiles = await storageService.listDirectory(codeFilePath);

            if (remoteFiles.length > 0) {
                let latestTimestamp = 0;

                trial.nodeExitResults = [];
                for (const fileName of remoteFiles) {
                    if (fileName.startsWith("code")) {
                        const fullName = storageService.joinPath(codeFilePath, fileName)
                        const fileContent = await storageService.readFileContent(fullName);

                        const match: RegExpMatchArray | null = fileContent.trim().match(/^-?(\d+)\s+(\d+)$/);
                        if (match !== null) {
                            const { 1: code, 2: timestamp } = match;
                            const intCode = parseInt(code, 10)
                            latestTimestamp = Math.max(latestTimestamp, parseInt(timestamp, 10));
                            if (intCode === 0) {
                                trial.nodeExitResults.push("SUCCEEDED");
                            } else {
                                trial.nodeExitResults.push("FAILED");
                            }
                        }
                    }
                }
            }
        }
    }

    public async startTrial(trial: TrialDetail): Promise<void> {
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
        const storageService = component.get<StorageService>(StorageService);
        const fileName = storageService.joinPath(trial.workingDirectory, generateParamFileName(form.hyperParameters))

        // Write file content ( parameter.cfg ) to working folders
        await storageService.save(form.hyperParameters.value, fileName);
    }

    private async sendCommand(commantType: string, data: any, environment: EnvironmentInformation): Promise<void> {
        let retryCount = 10;
        let fileName: string;
        let filePath: string = "";
        let findingName: boolean = true;
        const command = encodeCommand(commantType, JSON.stringify(data));
        const storageService = component.get<StorageService>(StorageService);
        const commandPath = storageService.joinPath(environment.workingFolder, `commands`);

        while (findingName) {
            fileName = `manager_command_${new Date().getTime()}.txt`;
            filePath = storageService.joinPath(commandPath, fileName);
            if (!await storageService.exists(filePath)) {
                findingName = false;
                break;
            }
            if (retryCount == 0) {
                throw new Error(`EnvironmentManager retry too many times to send command!`);
            }
            retryCount--;
            await delay(1);
        }

        // prevent to have imcomplete command, so save as temp name and then rename.
        await storageService.save(command.toString("utf8"), filePath);
    }
}
