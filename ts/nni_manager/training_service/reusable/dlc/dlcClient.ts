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
    public status: string;

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
        this.status = '';
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
        this.onMessage();
        this.log.debug(`on message`);
        this.monitorError(this.pythonShellClient, deferred);
        this.log.debug(`monitor submit`);
        const log = this.log;
        this.pythonShellClient.on('message', (message: any) => {
            const jobid = this.parseContent('job_id', message);
            if (jobid !== '') {
                log.debug(`reslove job_id ${jobid}`);
                deferred.resolve(jobid);
            }
        });
        return deferred.promise;
    }
    private onMessage(): void {
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        const log = this.log;
        this.pythonShellClient.on('message', (message: any) => {
            const status: string= this.parseContent('status', message);
            if (status.length > 0) {
                log.debug(`on message status: ${status}`)
                this.status = status;
                return;
            }
        });
    }
    public stop(): void {
        if (this.pythonShellClient === undefined) {
            this.log.debug(`python shell client not initialized!`);
            throw Error('python shell client not initialized!');
        }
        this.log.debug(`send stop`);
        this.pythonShellClient.send('stop');
    }

    public getTrackingUrl(): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.log.debug(`send tracking_url`);
        this.pythonShellClient.send('tracking_url');

        const log = this.log;
        this.pythonShellClient.on('message', (status: any) => {
            const trackingUrl = this.parseContent('tracking_url', status);
            if (trackingUrl !== '') {
                log.debug(`trackingUrl:${trackingUrl}`);
                deferred.resolve(trackingUrl);
            }
        });
        return deferred.promise;
    }

    public updateStatus(oldStatus: string): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();
        if (this.pythonShellClient === undefined) {
            throw Error('python shell client not initialized!');
        }
        this.pythonShellClient.send('update_status');
        if (this.status === '') {
            this.status = oldStatus;
        }
        this.log.debug(`update_status:${this.status}`);
        deferred.resolve(this.status);
        return deferred.promise;
    }

    // Monitor error information in dlc python shell client
    private monitorError(pythonShellClient: PythonShell, deferred: Deferred<any>): void {
        const log = this.log;
        pythonShellClient.on('error', function (error: any) {
            log.info(`error:${error}`);
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
