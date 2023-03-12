// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Manage SSH servers.
 **/

import path from 'node:path';

import type { WsChannel } from 'common/command_channel/websocket/channel';
import type { RemoteMachineConfig } from 'common/experimentConfig';
import { globals } from 'common/globals';
import { Logger, getLogger } from 'common/log';
import { runPythonScript } from 'common/pythonScript';
import type { EnvironmentInfo } from 'common/training_service_v3';
import { RemoteTrialKeeper } from 'common/trial_keeper/rpc';
import { Ssh } from './ssh';

export class Worker {
    private channel!: WsChannel;
    private channelUrl: string;
    private log: Logger;
    private ssh: Ssh;
    private trainingServiceId: string;
    private uploadDir!: string;

    public readonly channelId: string;
    public readonly config: RemoteMachineConfig;
    public env: EnvironmentInfo;
    public readonly trialKeeper: RemoteTrialKeeper;

    public get envId(): string {
        return `${this.trainingServiceId}-worker${this.channelId}`;
    }

    constructor(
            trainingServiceId: string,
            channelId: string,
            config: RemoteMachineConfig,
            channelUrl: string,
            enableGpuScheduling: boolean) {
        this.log = getLogger(`RemoteV3.Worker.${channelId}`);
        this.trainingServiceId = trainingServiceId;
        this.channelId = channelId;
        this.config = config;
        this.channelUrl = channelUrl;
        this.env = { id: this.envId };
        this.trialKeeper = new RemoteTrialKeeper(this.envId, 'remote', enableGpuScheduling);
        this.ssh = new Ssh(channelId, config);
    }

    public setChannel(channel: WsChannel): void {
        this.channel = channel;
        this.trialKeeper.setChannel(channel);
    }

    public async start(): Promise<void> {
        this.log.info('Initializing SSH worker', this.config.host);

        await this.ssh.connect();

        let python = await this.findPython();
        if (this.config.pythonPath) {
            python = await this.updatePath(python);
        }

        await this.ssh.run(`${python} -m pip install nni --upgrade`);  // FIXME: why upgrade???

        const remoteVersion = await this.ssh.run(`${python} -c "import nni ; print(nni.__version__)"`);
        const localVersion = await runPythonScript('import nni ; print(nni.__version__)');
        if (localVersion !== remoteVersion) {
            this.log.error(`NNI version mismatch. Local: ${localVersion} ; SSH server: ${remoteVersion}`);
        }

        this.uploadDir = await this.launchTrialKeeperDaemon(python);
        this.env = await this.trialKeeper.start();

        this.log.info(`Worker ${this.config.host} initialized`);
    }

    async stop(): Promise<void> {
        this.log.info('Stop worker', this.config.host);
        await this.trialKeeper.shutdown();
        this.channel.close('shutdown');
    }

    async upload(name: string, tar: string): Promise<void> {
        this.log.info('Uploading', name);
        const remotePath = path.join(this.uploadDir, `${name}.tgz`);
        await this.ssh.upload(tar, remotePath);
        await this.trialKeeper.unpackDirectory(name, remotePath);
        this.log.info('Upload success');
    }

    /**
     *  Find a usable python command.
     *
     *  If the user provides a `pythonPath` config, this python will be used to setup PATH env,
     *  and we will re-find python after that.
     *  In this case, python 2 is also acceptable.
     *
     *  Otherwise if there is no `pythonPath` config,
     *  the found python will be used for all tasks including user trials.
     **/
    private async findPython(): Promise<string> {
        const candidates = [];
        if (this.config.pythonPath) {
            if (!this.config.pythonPath.includes('\\')) {  // it might be a posix server
                candidates.push(this.config.pythonPath + '/python');
                candidates.push(this.config.pythonPath + '/python3');
            }
            if (!this.config.pythonPath.includes('/')) {  // it might be a windows server
                candidates.push(this.config.pythonPath + '\\python');
            }
            candidates.push(this.config.pythonPath);  // in case the user makes mistake
        }
        candidates.push('python');
        candidates.push('python3');

        let python2;

        for (const python of candidates) {
            const result = await this.ssh.exec(python + ' --version');
            if (result.code !== 0) {
                continue;
            }

            if (this.config.pythonPath) {
                if (result.stdout.startsWith('Python 2')) {
                    python2 = python;
                } else {
                    this.log.debug('Python for initializing:', python);
                    return python;
                }
            } else {
                this.log.info('Use following python command:', python);
                return python;
            }
        }

        if (python2) {
            this.log.warning('Cannot find python 3, using python 2 for initializing:', python2);
            return python2;
        }

        this.log.error('Cannot find python on SSH server');
        throw new Error(`Cannot find python on SSH server ${this.config.host}`);
    }

    /**
     *  Update PATH env using the given python interpreter.
     *  Return the new python command after setting up PATH.
     **/
    private async updatePath(python: string): Promise<string> {
        if (python === this.config.pythonPath) {
            this.log.error('python_path should be the directory rather than the executable, please check your config');
            return python;
        }

        const os = await this.ssh.run(`${python} -c "import sys ; print(sys.platform)"`);
        const envJson = await this.ssh.run(`${python} -c "import json,os ; print(json.dumps(dict(os.environ)))"`);
        const env = JSON.parse(envJson);

        const delimiter = (os === 'win32' ? ';' : ':');
        env['PATH'] = this.config.pythonPath + delimiter + env['PATH'];
        this.ssh.setEnv(env);

        for (const newPython of ['python', 'python3']) {
            const result = await this.ssh.exec(newPython + ' --version');
            if (result.code === 0 && !result.stdout.startsWith('Python 2')) {
                return newPython;
            }
        }
        this.log.error('Cannot find python after adding python_path', this.config.pythonPath);
        throw new Error(`Cannot find python after adding ${this.config.pythonPath} to PATH`);
    }

    /**
     *  Launch trial keeper daemon process.
     *  Return trial keeper's upload directory.
     **/
    private async launchTrialKeeperDaemon(python: string): Promise<string> {
        const prepareCommand = [
            python,
            '-m nni.tools.nni_manager_scripts.create_trial_keeper_dir',
            globals.args.experimentId,
            this.envId,
        ].join(' ');
        const trialKeeperDir = await this.ssh.run(prepareCommand);

        const launcherConfig = {
            environmentId: this.envId,
            experimentId: globals.args.experimentId,
            logLevel: globals.args.logLevel,
            managerCommandChannel: this.channelUrl,
            platform: 'remote',
        };
        await this.ssh.writeFile(path.join(trialKeeperDir, 'launcher_config.json'), JSON.stringify(launcherConfig));

        const launchCommand = `${python} -m nni.tools.nni_manager_scripts.launch_trial_keeper ${trialKeeperDir}`;
        const result = JSON.parse(await this.ssh.run(launchCommand));
        if (!result.success) {
            this.log.error('Failed to launch trial keeper daemon:', result);
            throw new Error('Failed to launch trial keeper daemon');
        }
        return result.uploadDirectory;
    }
}
