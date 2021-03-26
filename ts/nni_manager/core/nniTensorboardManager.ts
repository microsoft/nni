// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as cp from 'child_process';
import { ChildProcess } from 'child_process';
import { Deferred } from 'ts-deferred';

import * as component from '../common/component';
import { getLogger, Logger } from '../common/log';
import { getTunerProc, isAlive, killPid, uniqueString, mkDirPSync, getFreePort } from '../common/utils';
import { Manager } from '../common/manager';
import { TensorboardParams, TensorboardTaskStatus, TensorboardTaskInfo, TensorboardManager } from '../common/tensorboardManager';

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
        this.log = getLogger();
        this.tensorboardTaskMap = new Map<string, TensorboardTaskDetail>();
        this.setTensorboardVersion();
        this.nniManager = component.get(Manager);
    }

    public async startTensorboardTask(tensorboardParams: TensorboardParams): Promise<TensorboardTaskDetail> {
        const trialJobIds = tensorboardParams.trials;
        const trialJobIdList: string[] = [];
        const trialLogDirectoryList: string[] = [];
        await Promise.all(trialJobIds.split(',').map(async (trialJobId) => {
            const trialDataPath = await this.nniManager.getTrialOutputLocalPath(trialJobId);
            mkDirPSync(trialDataPath);
            trialJobIdList.push(trialJobId);
            trialLogDirectoryList.push(trialDataPath);
        }));
        this.log.info(`tensorboard: ${trialJobIdList} ${trialLogDirectoryList}`);
        return this.startTensorboardTaskProcess(trialJobIdList, trialLogDirectoryList);
    }

    private async startTensorboardTaskProcess(trialJobIdList: string[], trialLogDirectoryList: string[]): Promise<TensorboardTaskDetail> {
        const deferred = new Deferred<TensorboardTaskDetail>();
        try{
            const host = 'localhost';
            const port = await getFreePort(host, 6006, 65535);
            const command = this.getTensorboardStartCommand(trialJobIdList, trialLogDirectoryList, port);
            this.log.info(`tensorboard start command: ${command}`);
            const tensorboardProc: ChildProcess = getTunerProc(command, 'ignore', process.cwd(), process.env);
            const tensorboardTask = new TensorboardTaskDetail(uniqueString(5), 'RUNNING', trialJobIdList, trialLogDirectoryList);
            this.tensorboardTaskMap.set(tensorboardTask.id, tensorboardTask);
            tensorboardTask.pid = tensorboardProc.pid;
            tensorboardTask.port = `${port}`;
            this.log.info(`tensorboard task id: ${tensorboardTask.id}`);
            this.updateTensorboardTask(tensorboardTask.id);
            deferred.resolve(tensorboardTask);
        } catch (error) {
            deferred.reject(error);
        }
        return deferred.promise
    }

    private getTensorboardStartCommand(trialJobIdList: string[], trialLogDirectoryList: string[], port: number): string {
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
            trialJobIdList.forEach((trialJobId, idx) => {
                const realPath = fs.realpathSync(trialLogDirectoryList[idx]);
                logRealPaths.push(`${trialJobId}:${realPath}`);
            });
            const command = `tensorboard ${logdirCmd}=${logRealPaths.join(',')} --port=${port}`;
            return command;
        } catch (error){
            throw new Error(`${error.message}`);
        }
    }

    private setTensorboardVersion(): void {
        let command = 'python3 -m pip show tensorboard | grep Version:';
        if (process.platform === 'win32') {
            command = 'python -m pip show tensorboard | findstr Version:';
        }
        try {
            const tensorboardVersion = cp.execSync(command).toString().split(': ')[1].replace(/[\r\n]/g,"");
            this.tensorboardVersion = tensorboardVersion;
        } catch (error) {
            this.log.warning(`Tensorboard may not installed, if you want to use tensorboard, please check if tensorboard installed.`);
        }
    }

    public async getTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskDetail> {
        const deferred = new Deferred<TensorboardTaskDetail>();
        const tensorboardTask: TensorboardTaskDetail | undefined = this.tensorboardTaskMap.get(tensorboardTaskId);
        if (tensorboardTask === undefined) {
            deferred.reject(new Error('Tensorboard task not found'));
        }
        else{
            if (tensorboardTask.status !== 'STOPPED'){
                const alive: boolean = await isAlive(tensorboardTask.pid);
                if (!alive) {
                    this.setTensorboardTaskStatus(tensorboardTask, 'ERROR');
                }
            }
            deferred.resolve(tensorboardTask);
        }
        return deferred.promise;
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

    private async downloadDataFinished(tensorboardTask: TensorboardTaskDetail): Promise<void> {
        this.setTensorboardTaskStatus(tensorboardTask, 'RUNNING');
    }

    public async updateTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskInfo> {
        const deferred = new Deferred<TensorboardTaskDetail>();
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
            deferred.resolve(tensorboardTask);
        } else {
            deferred.reject(new Error('only tensorboard task with RUNNING or FAIL_DOWNLOAD_DATA can update data'));
        }
        return deferred.promise;
    }

    public async stopTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskInfo>{
        const deferred = new Deferred<TensorboardTaskDetail>();
        const tensorboardTask = await this.getTensorboardTask(tensorboardTaskId);
        if (['RUNNING', 'FAIL_DOWNLOAD_DATA'].includes(tensorboardTask.status)){
            this.setTensorboardTaskStatus(tensorboardTask, 'STOPPING');
            this.killTensorboardTaskProc(tensorboardTask);
            deferred.resolve(tensorboardTask);
        } else {
            deferred.reject(new Error('Only RUNNING FAIL_DOWNLOAD_DATA task can be stopped'));
        }
        return deferred.promise;
    }

    private async killTensorboardTaskProc(tensorboardTask: TensorboardTaskDetail): Promise<void> {
        if (['ERROR', 'STOPPED'].includes(tensorboardTask.status)) {
            return
        }
        const alive: boolean = await isAlive(tensorboardTask.pid);
        if (!alive) {
            this.setTensorboardTaskStatus(tensorboardTask, 'ERROR');
        } else {
            await killPid(tensorboardTask.pid);
            this.log.debug(`Tensorboard task ${tensorboardTask.id} stopped.`);
            this.setTensorboardTaskStatus(tensorboardTask, 'STOPPED');
        }
    }

    public async stop(): Promise<void> {
        this.tensorboardTaskMap.forEach(async (value) => {
            await this.killTensorboardTaskProc(value);
        });
        this.log.info('Tensorboard manager stopped.');
    }
}

export {
    NNITensorboardManager, TensorboardTaskDetail
};
