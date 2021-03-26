// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

interface TensorboardParams {
    trials: string;
}

type TensorboardTaskStatus = 'RUNNING' | 'DOWNLOADING_DATA' | 'STOPPING' | 'STOPPED' | 'ERROR' | 'FAIL_DOWNLOAD_DATA';

interface TensorboardTaskInfo {
    readonly id: string;
    readonly status: TensorboardTaskStatus;
    readonly trialJobIdList: string[];
    readonly trialLogDirectoryList: string[];
    readonly pid?: number;
    readonly port?: string;
}

abstract class TensorboardManager {
    public abstract startTensorboardTask(tensorboardParams: TensorboardParams): Promise<TensorboardTaskInfo>;
    public abstract getTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskInfo>;
    public abstract updateTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskInfo>;
    public abstract listTensorboardTasks(): Promise<TensorboardTaskInfo[]>;
    public abstract stopTensorboardTask(tensorboardTaskId: string): Promise<TensorboardTaskInfo>;
    public abstract stop(): Promise<void>;
}

export {
    TensorboardParams, TensorboardTaskStatus, TensorboardTaskInfo, TensorboardManager
}
