// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Deferred } from 'ts-deferred';
import { PythonShell } from 'python-shell';
import { getLogger, Logger } from '../../../common/log';

export class DlcClient {
    private log: Logger;
    public type: string;
    public image: string;
    public podCount: number;
    public ecsSpec: string;
    public region: string;
    // e.g., data1e6vg1tu0zi7, to generate it, go to 'Dataset Config' page of DLC
    //       create a NAS data and copy the 'DataSet ConfigurationID'
    public nasDataSourceId: string;
    public accessKeyId: string;
    public accessKeySecret: string;
    public experimentId: string;
    public environmentId: string;
    public userCommand: string;
    public pythonShellClient: undefined | PythonShell;
    public codeDir: string;

    constructor(
        type: string,
        image: string,
        podCount: number,
        experimentId: string,
        environmentId: string,
        ecsSpec: string,
        region: string,
        nasDataSourceId: string,
        accessKeyId: string,
        accessKeySecret: string,
        userCommand: string,
        codeDir: string,
        ) {
        this.log = getLogger('DlcClient');
        this.type = type;
        this.image = image;
        this.podCount = podCount;
        this.ecsSpec = ecsSpec;
        this.image = image;
        this.region = region;
        this.nasDataSourceId = nasDataSourceId;
        this.accessKeyId = accessKeyId;
        this.accessKeySecret = accessKeySecret
        this.experimentId = experimentId;
        this.environmentId = environmentId;
        this.userCommand = userCommand;
        this.codeDir = codeDir;
    }

    private getScript(): string[] {
        const script: string[] = [];
        script.push(
            `python ./config/dlc/dlcUtil.py --type ${this.type} --image ${this.image} --pod_count ${this.podCount} ` +
            `--ecs_spec ${this.ecsSpec} --experiment_name nni_exp_${this.experimentId} ` +
            `--region ${this.region} --nas_data_source_id ${this.nasDataSourceId} --access_key_id ${this.accessKeyId} ` + 
            `--access_key_secret ${this.accessKeySecret} --user_command "${this.userCommand}" --script_dir ${this.codeDir}` );
        return script;
    }

    public submit(): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        this.pythonShellClient = new PythonShell('dlcUtil.py', {
            scriptPath: './config/dlc',
            pythonPath: 'python',
            pythonOptions: ['-u'], // get print results in real-time
            args: [
                '--type', this.type,
                '--image', this.image,
                '--pod_count', String(this.podCount),
                '--ecs_spec', this.ecsSpec,
                '--region', this.region,
                '--nas_data_source_id', this.nasDataSourceId,
                '--access_key_id', this.accessKeyId,
                '--access_key_secret', this.accessKeySecret,
                '--experiment_name', `nni_exp_${this.experimentId}_env_${this.environmentId}`,
                '--script_dir', this.codeDir,
                '--user_command', this.userCommand,
              ]
        });
        this.log.debug(this.pythonShellClient.command);
        this.pythonShellClient.on('message', function (envId: any) {
            // received a message sent from the Python script (a simple "print" statement)
            deferred.resolve(envId);
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;

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
        var log = this.log;
        this.pythonShellClient.on('message', (status: any) => {
            log.debug(`updateStatus: message  ${status}`);
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
        this.log.debug(`command:${message}`);
        this.pythonShellClient.send(`command:${message}`);
    }

    public receiveCommand(): Promise<any> {
        const deferred: Deferred<any> = new Deferred<any>();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('receive');
        var log = this.log;
        this.pythonShellClient.on('message', (command: any) => {
            log.debug(`message*** ${command}`);
            const message = this.parseContent('receive', command);
            if (message !== '') {
                deferred.resolve(JSON.parse(message))
            }
        });
        this.monitorError(this.pythonShellClient, deferred);
        return deferred.promise;
    }
    
    // Monitor error information in dlc python shell client
    private monitorError(pythonShellClient: PythonShell, deferred: Deferred<any>): void {
        var log = this.log;
        pythonShellClient.on('error', function (error: any) {
            log.debug(`error*** ${error}`);
            deferred.reject(error);
        });
        pythonShellClient.on('close', function (error: any) {
            log.debug(`close*** ${error}`);
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
