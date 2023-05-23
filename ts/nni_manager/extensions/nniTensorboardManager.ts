// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'fs';
import cp from 'child_process';
import path from 'path';
import { ChildProcess } from 'child_process';

import { getLogger, Logger } from '../common/log';
import { getTunerProc, isAlive, uniqueString, mkDirPSync, getFreePort } from '../common/utils';
import { Manager } from '../common/manager';
import { TensorboardParams, TensorboardTaskStatus, TensorboardTaskInfo, TensorboardManager } from '../common/tensorboardManager';
import { globals } from 'common/globals';
import { IocShim } from 'common/ioc_shim';

class TensorboardTaskDetail implements TensorboardTaskInfo {
    public id: string;
    public status: TensorboardTaskStatus;
    public trialJobIdList: string[];
    public trialLogDirectoryList: string[];
    public pid?: number;
    public port?: string;

    constructor(id: string, status: TensorboardTaskStatus, trialJobIdList: string[], trialLogDirectoryList: string[]) {
        this.id = id;
        this.status = status;
        this.trialJobIdList = trialJobIdList;
        this.trialLogDirectoryList = trialLogDirectoryList;
    }
}

class NNITensorboardManager implements TensorboardManager {
    private log: Logger;
    private tensorboardTaskMap: Map<string, TensorboardTaskDetail>;
    private tensorboardVersion: string | undefined;
    private nniManager: Manager;

    constructor() {
        this.log = getLogger('NNITensorboardManager');
        this.tensorboardTaskMap = new Map<string, TensorboardTaskDetail>();
        this.setTensorboardVersion();
        this.nniManager = IocShim.get(Manager);
    }

    public async startTensorboardTask(tensorboardParams: TensorboardParams): Promise<TensorboardTaskDetail> {
        const trialJobIds = tensorboardParams.trials;
        const trialJobIdList: string[] = [];
        const trialLogDirectoryList: string[] = [];
        await Promise.all(trialJobIds.split(',').map(async (trialJobId) => {
            const trialTensorboardDataPath = path.join(await this.nniManager.getTrialOutputLocalPath(trialJobId), 'tensorboard');
            mkDirPSync(trialTensorboardDataPath);
            trialJobIdList.push(trialJobId);
            trialLogDirectoryList.push(trialTensorboardDataPath);
        }));
        this.log.info(`tensorboard: ${trialJobIdList} ${trialLogDirectoryList}`);
        return await this.startTensorboardTaskProcess(trialJobIdList, trialLogDirectoryList);
    }

    private async startTensorboardTaskProcess(trialJobIdList: string[], trialLogDirectoryList: string[]): Promise<TensorboardTaskDetail> {
        const host = 'localhost';
        const port = await getFreePort(host, 6006, 65535);
        const command = await this.getTensorboardStartCommand(trialJobIdList, trialLogDirectoryList, port);
        this.log.info(`tensorboard start command: ${command}`);
        const tensorboardTask = new TensorboardTaskDetail(uniqueString(5), 'RUNNING', trialJobIdList, trialLogDirectoryList);
        this.tensorboardTaskMap.set(tensorboardTask.id, tensorboardTask);

        const tensorboardProc: ChildProcess = getTunerProc(command, 'ignore', process.cwd(), process.env, true, true);
        tensorboardProc.on('error', async (error) => {
            this.log.error(error);
            const alive: boolean = await isAlive(tensorboardProc.pid);
            if (alive) {
                process.kill(-tensorboardProc.pid!);
            }
            this.setTensorboardTaskStatus(tensorboardTask, 'ERROR');
        });
        tensorboardTask.pid = tensorboardProc.pid;

        tensorboardTask.port = `${port}`;
        this.log.info(`tensorboard task id: ${tensorboardTask.id}`);
        this.updateTensorboardTask(tensorboardTask.id);
        return tensorboardTask;
    }

    private async getTensorboardStartCommand(trialJobIdList: string[], trialLogDirectoryList: string[], port: number): Promise<string[]> {
        if (this.tensorboardVersion === undefined) {
            this.setTensorboardVersion();
            if (this.tensorboardVersion === undefined) {
                throw new Error(`Tensorboard may not installed, if you want to use tensorboard, please check if tensorboard installed.`);
            }
        }
        if (trialJobIdList.length !== trialLogDirectoryList.length) {
            throw new Error('trial list length does not match');
        }
        if (trialJobIdList.length === 0) {
            throw new Error('trial list length is 0');
        }
        let logdirCmd = '--logdir';
        if (this.tensorboardVersion >= '2.0') {
            logdirCmd = '--bind_all --logdir_spec'
        }
        try {
            const logRealPaths: string[] = [];
            for (const idx in trialJobIdList) {
                const realPath = fs.realpathSync(trialLogDirectoryList[idx]);
                const trialJob = await this.nniManager.getTrialJob(trialJobIdList[idx]);
                logRealPaths.push(`${trialJob.sequenceId}-${trialJobIdList[idx]}:${realPath}`);
            }
            return [ 'tensorboard', `${logdirCmd}=${logRealPaths.join(',')}`, `--port=${port}` ];
        } catch (error) {
            throw new Error(`${(error as any).message}`);
        }
    }

    private setTensorboardVersion(): void {
        let command = `${globals.args.pythonInterpreter} -c 'import tensorboard ; print(tensorboard.__version__)' 2>&1`;
        if (process.platform === 'win32') {
            command = `python -c "import tensorboard ; print(tensorboard.__version__)" 2>&1`;
        }
        try {
            const tensorboardVersion = cp.execSync(command).toString();
            if (/\d+(.\d+)*/.test(tensorboardVersion)) {
                this.tensorboardVersion = tensorboardVersion;
            }
        } catch (error) {
            this.log.warning(`Tensorboard may not installed, if you want to use tensorboard, please check if tensorboard installed.`);
        }
    }

    public async getTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskDetail> {
        const tensorboardTask: TensorboardTaskDetail | undefined = this.tensorboardTaskMap.get(tensorboardTaskId);
        if (tensorboardTask === undefined) {
            throw new Error('Tensorboard task not found');
        }
        else{
            if (tensorboardTask.status !== 'STOPPED'){
                const alive: boolean = await isAlive(tensorboardTask.pid);
                if (!alive) {
                    this.setTensorboardTaskStatus(tensorboardTask, 'ERROR');
                }
            }
            return tensorboardTask;
        }
    }

    public async listTensorboardTasks(): Promise<TensorboardTaskDetail[]> {
        const result: TensorboardTaskDetail[] = [];
        this.tensorboardTaskMap.forEach((value) => {
            result.push(value);
        });
        return result;
    }

    private setTensorboardTaskStatus(tensorboardTask: TensorboardTaskDetail, newStatus: TensorboardTaskStatus): void {
        if (tensorboardTask.status !== newStatus) {
            const oldStatus = tensorboardTask.status;
            tensorboardTask.status = newStatus;
            this.log.info(`tensorboardTask ${tensorboardTask.id} status update: ${oldStatus} to ${tensorboardTask.status}`);
        }
    }

    private downloadDataFinished(tensorboardTask: TensorboardTaskDetail): void {
        this.setTensorboardTaskStatus(tensorboardTask, 'RUNNING');
    }

    public async updateTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskInfo> {
        const tensorboardTask: TensorboardTaskDetail = await this.getTensorboardTask(tensorboardTaskId);
        if (['RUNNING', 'FAIL_DOWNLOAD_DATA'].includes(tensorboardTask.status)){
            this.setTensorboardTaskStatus(tensorboardTask, 'DOWNLOADING_DATA');
            Promise.all(tensorboardTask.trialJobIdList.map((trialJobId) => {
                this.nniManager.fetchTrialOutput(trialJobId, 'tensorboard');
            })).then(() => {
                this.downloadDataFinished(tensorboardTask);
            }).catch((error: Error) => {
                this.setTensorboardTaskStatus(tensorboardTask, 'FAIL_DOWNLOAD_DATA');
                this.log.error(`${error.message}`);
            });
            return tensorboardTask;
        } else {
            throw new Error('only tensorboard task with RUNNING or FAIL_DOWNLOAD_DATA can update data');
        }
    }

    public async stopTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskInfo> {
        const tensorboardTask = await this.getTensorboardTask(tensorboardTaskId);
        if (['RUNNING', 'FAIL_DOWNLOAD_DATA'].includes(tensorboardTask.status)){
            this.killTensorboardTaskProc(tensorboardTask);
            return tensorboardTask;
        } else {
            throw new Error('Only RUNNING FAIL_DOWNLOAD_DATA task can be stopped');
        }
    }

    private async killTensorboardTaskProc(tensorboardTask: TensorboardTaskDetail): Promise<void> {
        if (['ERROR', 'STOPPED'].includes(tensorboardTask.status)) {
            return
        }
        const alive: boolean = await isAlive(tensorboardTask.pid);
        if (!alive) {
            this.setTensorboardTaskStatus(tensorboardTask, 'ERROR');
        } else {
            this.setTensorboardTaskStatus(tensorboardTask, 'STOPPING');
            if (tensorboardTask.pid) {
                process.kill(-tensorboardTask.pid);
            }
            this.log.debug(`Tensorboard task ${tensorboardTask.id} stopped.`);
            this.setTensorboardTaskStatus(tensorboardTask, 'STOPPED');
            this.tensorboardTaskMap.delete(tensorboardTask.id);
        }
    }

    public async stopAllTensorboardTask(): Promise<void> {
        this.log.info('Forced stopping all tensorboard task.')
        for (const task of this.tensorboardTaskMap) {
            await this.killTensorboardTaskProc(task[1]);
        }
        this.log.info('All tensorboard task stopped.')
    }

    public async stop(): Promise<void> {
        await this.stopAllTensorboardTask();
        this.log.info('Tensorboard manager stopped.');
    }
}

export {
    NNITensorboardManager, TensorboardTaskDetail
};
