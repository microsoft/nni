// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import type { TensorboardManager, TensorboardParams, TensorboardTaskInfo } from 'common/tensorboardManager';

const taskInfo: TensorboardTaskInfo = {
    id: 'ID',
    status: 'RUNNING',
    trialJobIdList: [],
    trialLogDirectoryList: [],
    pid: undefined,
    port: undefined,
};

export class MockTensorboardManager implements TensorboardManager {
    public async startTensorboardTask(_tensorboardParams: TensorboardParams): Promise<TensorboardTaskInfo> {
        return taskInfo;
    }

    public async getTensorboardTask(_tensorboardTaskId: string): Promise<TensorboardTaskInfo> {
        return taskInfo;
    }

    public async updateTensorboardTask(_tensorboardTaskId: string): Promise<TensorboardTaskInfo> {
        return taskInfo;
    }

    public async listTensorboardTasks(): Promise<TensorboardTaskInfo[]> {
        return [ taskInfo ];
    }

    public async stopTensorboardTask(_tensorboardTaskId: string): Promise<TensorboardTaskInfo> {
        return taskInfo;
    }

    public async stopAllTensorboardTask(): Promise<void> {
        return;
    }

    public async stop(): Promise<void> {
        return;
    }
}
