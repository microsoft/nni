// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

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
    public computeTarget: string;

    constructor(
        subscriptionId: string,
        resourceGroup: string,
        workspaceName: string,
        experimentId: string,
        computeTarget: string,
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
        this.computeTarget = computeTarget;
    }

    public submit(): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        this.pythonShellClient = new PythonShell('amlUtil.py', {
            scriptPath: './config/aml',
            pythonPath: process.platform === 'win32' ? 'python' : 'python3',
            pythonOptions: ['-u'], // get print results in real-time
            args: [
                '--subscription_id', this.subscriptionId,
                '--resource_group', this.resourceGroup,
                '--workspace_name', this.workspaceName,
                '--compute_target', this.computeTarget,
                '--docker_image', this.image,
                '--experiment_name', `nni_exp_${this.experimentId}`,
                '--script_dir', this.codeDir,
                '--script_name', this.scriptName
              ]
        });
        this.pythonShellClient.on('message', function (envId: any) {
            // received a message sent from the Python script (a simple "print" statement)
            deferred.resolve(envId);
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;
    }

    public stop(): void {
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
        this.pythonShellClient.on('message', (status: any) => {
            const trackingUrl = this.parseContent('tracking_url', status);
            if (trackingUrl !== '') {
                deferred.resolve(trackingUrl);
            }
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;
    }

    public updateStatus(oldStatus: string): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('update_status');
        this.pythonShellClient.on('message', (status: any) => {
            let newStatus = this.parseContent('status', status);
            if (newStatus === '') {
                newStatus = oldStatus;
            }
            deferred.resolve(newStatus);
        });
        this.monitorError(this.pythonShellClient, deferred);
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
        this.pythonShellClient.on('message', (command: any) => {
            const message = this.parseContent('receive', command);
            if (message !== '') {
                deferred.resolve(JSON.parse(message))
            }
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;
    }
    
    // Monitor error information in aml python shell client
    private monitorError(pythonShellClient: PythonShell, deferred: Deferred<any>): void {
        pythonShellClient.on('error', function (error: any) {
            deferred.reject(error);
        });
        pythonShellClient.on('close', function (error: any) {
            deferred.reject(error);
        });
    }
    
    // Parse command content, command format is {head}:{content}
    public parseContent(head: string, command: string): string {
        const items = command.split(':');
        if (items[0] === head) {
            return command.slice(head.length + 1);
        }
        return '';
    }
}
