// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { Deferred } from 'ts-deferred';
import { PythonShell } from 'python-shell';
import { getLogger, Logger } from 'common/log';

export class DlcClient {
    private log: Logger;
    public type: string;
    public image: string;
    public jobType: string;
    public podCount: number;
    public ecsSpec: string;
    public region: string;
    public workspaceId: string;
    // e.g., data1e6vg1tu0zi7, to generate it, go to 'Dataset Config' page of DLC
    //       create a NAS data and copy the 'DataSet ConfigurationID'
    public nasDataSourceId: string;
    public ossDataSourceId: string;
    public accessKeyId: string;
    public accessKeySecret: string;
    public experimentId: string;
    public environmentId: string;
    public userCommand: string;
    // dlcUtil exception log dir
    public logDir: string;
    public pythonShellClient: undefined | PythonShell;

    constructor(
        type: string,
        image: string,
        jobType: string,
        podCount: number,
        experimentId: string,
        environmentId: string,
        ecsSpec: string,
        region: string,
        workspaceId: string,
        nasDataSourceId: string,
        accessKeyId: string,
        accessKeySecret: string,
        userCommand: string,
        logDir: string,
        ossDataSourceId?: string,
        ) {
        this.log = getLogger('DlcClient');
        this.type = type;
        this.image = image;
        this.jobType = jobType;
        this.podCount = podCount;
        this.ecsSpec = ecsSpec;
        this.image = image;
        this.region = region;
        this.workspaceId = workspaceId;
        this.nasDataSourceId = nasDataSourceId;
        if (ossDataSourceId !== undefined) {
            this.ossDataSourceId = ossDataSourceId;
        } else {
            this.ossDataSourceId = '';
        }
        this.accessKeyId = accessKeyId;
        this.accessKeySecret = accessKeySecret
        this.experimentId = experimentId;
        this.environmentId = environmentId;
        this.userCommand = userCommand;
        this.logDir = logDir;
    }

    public submit(): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        this.pythonShellClient = new PythonShell('dlcUtil.py', {
            scriptPath: './config/dlc',
            pythonPath: 'python3',
            pythonOptions: ['-u'], // get print results in real-time
            args: [
                '--type', this.type,
                '--image', this.image,
                '--job_type', this.jobType,
                '--pod_count', String(this.podCount),
                '--ecs_spec', this.ecsSpec,
                '--region', this.region,
                '--workspace_id', this.workspaceId,
                '--nas_data_source_id', this.nasDataSourceId,
                '--oss_data_source_id', this.ossDataSourceId,
                '--access_key_id', this.accessKeyId,
                '--access_key_secret', this.accessKeySecret,
                '--experiment_name', `nni_exp_${this.experimentId}_env_${this.environmentId}`,
                '--user_command', this.userCommand,
                '--log_dir', this.logDir,
              ]
        });
        this.log.debug(this.pythonShellClient.command);
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
        this.log.debug(`command:${message}`);
        this.pythonShellClient.send(`command:${message}`);
    }

    public receiveCommand(): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
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
    
    // Monitor error information in dlc python shell client
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
