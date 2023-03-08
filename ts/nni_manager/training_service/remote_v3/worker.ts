// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'node:fs'; 
import path from 'node:path';

import type { ConnectConfig } from 'ssh2';

import type { WsChannel } from 'common/command_channel/websocket/channel';
import type { RemoteMachineConfig } from 'common/experimentConfig';
import { globals } from 'common/globals';
import { Logger, getLogger } from 'common/log';
import type { EnvironmentInfo } from 'common/training_service_v3';
import { RemoteTrialKeeper } from 'common/trial_keeper/rpc';
import { Ssh } from './ssh';

export class Worker {
    trainingServiceId: string;
    channelId: string;
    channelUrl: string;
    trialKeeper: RemoteTrialKeeper;
    ssh: Ssh;
    launchResult!: any;
    log: Logger;
    config: RemoteMachineConfig;

    public get envId(): string {
        return `${this.trainingServiceId}-worker${this.channelId}`;
    }

    constructor(trainingServiceId: string, channelId: string, channelUrl: string, config: RemoteMachineConfig, enableGpu: boolean) {
        this.log = getLogger('Worker.TODO');
        this.trainingServiceId = trainingServiceId;
        this.channelId = channelId;
        this.channelUrl = channelUrl;
        this.trialKeeper = new RemoteTrialKeeper(this.envId, 'remote', enableGpu);
        this.config = config;

        const sshConfig: ConnectConfig = {
            host: config.host,
            port: config.port,
            username: config.user,
            password: config.password,
        };
        if (config.sshKeyFile) {
            sshConfig.privateKey = fs.readFileSync(config.sshKeyFile, { encoding: 'utf8' });
            sshConfig.passphrase = config.sshPassphrase;
        }

        this.ssh = new Ssh(sshConfig);
    }

    getEnv(): EnvironmentInfo {
        return { id: this.envId };
    }

    public setChannel(channel: WsChannel): void {
        this.trialKeeper.setChannel(channel);
    }

    private async findPython(): Promise<string> {
        const candidates = [];
        if (this.config.pythonPath) {
            if (!this.config.pythonPath.includes('\\')) {
                candidates.push(this.config.pythonPath + '/python');
                candidates.push(this.config.pythonPath + '/python3');
            }
            if (!this.config.pythonPath.includes('/')) {
                candidates.push(this.config.pythonPath + '\\python');
            }
            candidates.push(this.config.pythonPath);  // in case the user makes mistake
        }
        candidates.push('python');
        candidates.push('python3');

        let python2;

        for (const python of candidates) {
            const result = await this.ssh.exec(python + ' --version');
            if (result.code === 0) {
                if (result.stdout.startsWith('Python 2.')) {
                    python2 = python;
                } else {
                    this.log.debug('Python command chosen for initializing:', python);
                    return python;
                }
            }
        }

        if (this.config.pythonPath && python2) {
            this.log.warning('Cannot find python 3, using python 2 for initializing:', python2);
            return python2;
        }

        this.log.error('Cannot find python on SSH server');
        throw new Error(`Cannot find python on server ${this.config.host}`);
    }

    private async setPythonPath(python: string): Promise<string> {
        if (python === this.config.pythonPath) {
            this.log.error('python_path should be the directory rather than the executable, please check your config');
            return python;
        }

        const osCmd = `${python} -c "import sys ; print(sys.platform)"`;
        const osResult = await this.ssh.exec(osCmd);
        if (!osResult.stdout) {
            this.log.error('Failed to detect OS', osResult);
            throw new Error(`Failed to detect OS for server ${this.config.host}`);
        }
        const os = osResult.stdout.trim();

        const envCmd = `${python} -c "import json,os ; print(json.dumps(dict(os.environ)))"`
        const envResult = await this.ssh.exec(envCmd);
        if (!envResult.stdout) {
            this.log.error('Failed to get env', envResult);
            throw new Error(`Failed to get environment variables for server ${this.config.host}`);
        }
        const env = JSON.parse(envResult.stdout);

        const delimiter = (os === 'win32' ? ';' : ':');
        env['PATH'] = this.config.pythonPath + delimiter + env['PATH'];
        this.ssh.setEnv(env);

        for (const newPython of ['python', 'python3']) {
            const result = await this.ssh.exec(newPython + ' --version');
            if (result.code === 0 && !result.stdout.startsWith('Python 2.')) {
                return python;
            }
        }
        this.log.error('Cannot find python after setting pythonPath');
        throw new Error(`Cannot find python after adding ${this.config.pythonPath} to PATH`);
    }

    public async start(): Promise<void> {
        await this.ssh.connect();

        let python = await this.findPython();
        if (this.config.pythonPath) {
            python = await this.setPythonPath(python);
        }

        await this.ssh.run(`${python} -m pip install nni --upgrade`);  // FIXME: upgrade???

        // todo: check version

        const cmd = `${python} -m nni.tools.nni_manager_scripts.create_tmp_dir ${globals.args.experimentId} ${this.envId}`;
        const tmpDir = await this.ssh.run(cmd);

        const config = {
            experimentId: globals.args.experimentId,
            logLevel: globals.args.logLevel,
            platform: 'remote',
            environmentId: this.envId,
            managerCommandChannel: this.channelUrl,
        };
        await this.ssh.writeFile(path.join(tmpDir, 'config.json'), JSON.stringify(config));

        const result = await this.ssh.run(`${python} -m nni.tools.nni_manager_scripts.launch_trial_keeper ${tmpDir}`);
        this.launchResult = JSON.parse(result);

        await this.trialKeeper.start();
    }

    async stop(): Promise<void> {
        await this.trialKeeper.shutdown();
    }

    async upload(name: string, tar: string): Promise<void> {
        const remotePath = path.join(this.launchResult.envDir, 'upload', `${name}.tgz`);
        await this.ssh.upload(tar, remotePath);
        await this.trialKeeper.unpackDirectory(name, remotePath);
    }
}
