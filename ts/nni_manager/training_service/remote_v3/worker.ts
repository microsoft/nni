// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Manage SSH servers.
 **/

import fs from 'node:fs/promises';
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
    private python!: string;
    private remoteTrialKeeperDir!: string;
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
            enableGpuScheduling: boolean
        ) {

        this.log = getLogger(`RemoteV3.Worker.${channelId}`);
        this.trainingServiceId = trainingServiceId;
        this.channelId = channelId;
        this.config = config;
        this.channelUrl = channelUrl;
        this.env = { id: this.envId, host: config.host };
        this.trialKeeper = new RemoteTrialKeeper(this.envId, 'remote', enableGpuScheduling);
        this.ssh = new Ssh(channelId, config);
    }

    public setChannel(channel: WsChannel): void {
        this.channel = channel;
        this.trialKeeper.setChannel(channel);
        channel.onLost(async () => {
            if (!await this.checkAlive()) {
                this.log.error('Trial keeper failed');
                channel.terminate('Trial keeper failed');  // MARK
            }
        });
    }

    public async start(): Promise<void> {
        this.log.info('Initializing SSH worker', this.config.host);

        await this.ssh.connect();

        this.python = await this.findPython();

        this.log.info('Installing nni and dependencies...');
        await this.ssh.run(`${this.python} -m pip install nni --upgrade`);  // FIXME: why upgrade???

        const remoteVersion = await this.ssh.run(`${this.python} -c "import nni ; print(nni.__version__)"`);
        this.log.info(`Installed nni v${remoteVersion}`);

        const localVersion = await runPythonScript('import nni ; print(nni.__version__)');
        if (localVersion.trim() !== remoteVersion.trim()) {
            this.log.error(`NNI version mismatch. Local: ${localVersion.trim()} ; SSH server: ${remoteVersion}`);
        }

        await this.launchTrialKeeperDaemon();
        this.env = await this.trialKeeper.start();
        this.env['host'] = this.config.host;

        this.log.info(`Worker ${this.config.host} initialized`);
    }

    public async stop(): Promise<void> {
        this.log.info('Stop worker', this.config.host);
        await this.trialKeeper.shutdown();
        this.channel.close('shutdown');
    }

    public async upload(name: string, tar: string): Promise<void> {
        this.log.info('Uploading', name);
        const remotePath = path.join(this.uploadDir, `${name}.tgz`);
        await this.ssh.upload(tar, remotePath);
        await this.trialKeeper.unpackDirectory(name, remotePath);
        this.log.info('Upload success');
    }

    private async findPython(): Promise<string> {
        let python = await this.findInitPython();
        if (this.config.pythonPath) {
            python = await this.updatePath(python);
        }
        return python;
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
    private async findInitPython(): Promise<string> {
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
        this.ssh.setPath(this.config.pythonPath + delimiter + env['PATH']);

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
    private async launchTrialKeeperDaemon(): Promise<void> {
        const prepareCommand = [
            this.python,
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
        this.log.debug('Trial keeper launcher config:', launcherConfig);
        await this.ssh.writeFile(path.join(trialKeeperDir, 'launcher_config.json'), JSON.stringify(launcherConfig));

        const launchCommand = `${this.python} -m nni.tools.nni_manager_scripts.launch_trial_keeper ${trialKeeperDir}`;
        const result = JSON.parse(await this.ssh.run(launchCommand));
        if (!result.success) {
            this.log.error('Failed to launch trial keeper daemon:', result);
            throw new Error('Failed to launch trial keeper daemon');
        }

        this.uploadDir = result.uploadDirectory;
        this.remoteTrialKeeperDir = result.trialKeeperDirectory;
    }

    private async checkAlive(): Promise<boolean> {
        try {
            const command = [
                this.python,
                '-m',
                'nni.tools.nni_manager_scripts.check_trial_keeper_alive',
                this.remoteTrialKeeperDir
            ].join(' ');
            const alive = JSON.parse(await this.ssh.run(command));

            if (alive.alive) {
                return true;
            } else {
                this.log.error('Trial keeper not alive:', alive);
                return false;
            }

        } catch (error) {
            this.log.error('Failed to check trail keeper status:', error);
            return false;
        }
    }

    public async downloadTrialLog(trialId: string): Promise<string> {
        this.log.debug('Downloading trial log:', trialId);

        // FIXME: hack
        const localDir = path.join(globals.paths.experimentRoot, 'trials', trialId);
        const remoteDir = path.join(path.dirname(this.uploadDir), 'trials', trialId);

        await fs.mkdir(localDir, { recursive: true });

        for (const file of ['trial.log', 'trial.stdout', 'trial.stderr']) {
            try {
                await this.ssh.download(path.join(remoteDir, file), path.join(localDir, file));
            } catch (error) {
                this.log.warning(`Cannot download ${file} of ${trialId}`);
            }
        }

        return localDir;
    }

    /*  used to debug pipeline. re-enable it when we support log collection

    private async downloadTrialKeeperLog(): Promise<void> {
        this.log.debug('Downloading trial keeper log');

        const localDir = path.join(globals.paths.experimentRoot, 'environments', this.envId, 'trial_keeper_log');
        await fs.mkdir(localDir, { recursive: true });

        // fixme
        const remoteDir = path.join(path.dirname(this.uploadDir), 'trial_keeper');

        await Promise.all([
            this.ssh.download(path.join(remoteDir, 'trial_keeper.log'), path.join(localDir, 'trial_keeper.log')),
            this.ssh.download(path.join(remoteDir, 'trial_keeper.stdout'), path.join(localDir, 'trial_keeper.stdout')),
            this.ssh.download(path.join(remoteDir, 'trial_keeper.stderr'), path.join(localDir, 'trial_keeper.stderr')),
        ]);

        const log = await fs.readFile(path.join(localDir, 'trial_keeper.log'), { encoding: 'utf8' });
        console.error('## Trial keeper log:');
        console.error(log);

        const stderr = await fs.readFile(path.join(localDir, 'trial_keeper.stderr'), { encoding: 'utf8' });
        console.error('## Trial keeper stderr:');
        console.error(stderr);
    }
    */
}
