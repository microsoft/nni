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
import { PythonShell } from 'python-shell';

export class AMLClient {
    public subscriptionId: string;
    public resourceGroup: string;
    public workspaceName: string;
    public experimentId: string;
    public image: string;
    public scriptName: string;
    public pythonShellClient: undefined | PythonShell;
    public codeDir: string;
    public computerTarget: string;
    private readonly NNI_METRICS_PATTERN: string = `NNISDK_MEb'(?<metrics>.*?)'`;

    constructor(
        subscriptionId: string,
        resourceGroup: string,
        workspaceName: string,
        experimentId: string,
        computerTarget: string,
        image: string,
        scriptName: string,
        codeDir: string,
        ) {
        this.subscriptionId = subscriptionId;
        this.resourceGroup = resourceGroup;
        this.workspaceName = workspaceName;
        this.experimentId = experimentId;
        this.image = image;
        this.scriptName = scriptName;
        this.codeDir = codeDir;
        this.computerTarget = computerTarget;
    }

    public async submit(): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        this.pythonShellClient = new PythonShell('amlUtil.py', {
            scriptPath: './config/aml',
            pythonOptions: ['-u'], // get print results in real-time
            args: [
                '--subscription_id', this.subscriptionId,
                '--resource_group', this.resourceGroup,
                '--workspace_name', this.workspaceName,
                '--computer_target', this.computerTarget,
                '--docker_image', this.image,
                '--experiment_name', `nni_exp_${this.experimentId}`,
                '--code_dir', this.codeDir,
                '--script', this.scriptName
              ]
        });
        this.pythonShellClient.on('message', function (envId: any) {
            // received a message sent from the Python script (a simple "print" statement)
            deferred.resolve(envId);
        });
        return deferred.promise;
    }

    public stop() {
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('stop');
    }

    public getTrackingUrl(): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('tracking_url');
        let trackingUrl = '';
        this.pythonShellClient.on('message', function (status: any) {
            let items = status.split(':');
            if (items[0] === 'tracking_url') {
                trackingUrl = items.splice(1, items.length).join('')
            }
            deferred.resolve(trackingUrl);
        });
        return deferred.promise;
    }

    public updateStatus(oldStatus: string): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        let newStatus = oldStatus;
        this.pythonShellClient.send('update_status');
        this.pythonShellClient.on('message', function (status: any) {
            let items = status.split(':');
            if (items[0] === 'status') {
                newStatus = items.splice(1, items.length).join('')
            }
            deferred.resolve(newStatus);
        });
        return deferred.promise;
    }

    public sendCommand(message: string): void {
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send(`command:${message}`);
    }

    public receiveCommand(): Promise<any> {
        const deferred: Deferred<any> = new Deferred<any>();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('receive');
        this.pythonShellClient.on('message', function (command: any) {
            let items = command.split(':')
            if (items[0] === 'receive') {
                deferred.resolve(JSON.parse(command.slice(8)))
            }
        });
        return deferred.promise;
    }
}