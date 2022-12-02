// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';

import type { TrialKeeper } from 'common/trial_keeper/keeper';
import type { GpuSystemInfo } from 'common/trial_keeper/task_scheduler/collect_info';
import { TaskScheduler, UnitTestHelpers as Helpers } from 'common/trial_keeper/task_scheduler/scheduler';

let scheduler: TaskScheduler;

const gpuInfo: GpuSystemInfo = {
    success: true,
    gpuNumber: 3,
    gpus: [
        { index: 0, gpuCoreUtilization: 0, gpuMemoryUtilization: 0 },
        { index: 1, gpuCoreUtilization: 0, gpuMemoryUtilization: 0 },
        { index: 2, gpuCoreUtilization: 0, gpuMemoryUtilization: 0 },
    ],
    processes: [],
    timestamp: 0,
}

describe('## gpu scheduler ##', () => {
    it('init', () => testInit());

    it('schedule small trials', () => testScheduleSmall());
    it('release all', () => testReleaseSmall());
    it('schedule large trials', () => testScheduleLarge());
    it('release one by one', () => testReleaseLarge());
    it('schedule hybrid', () => testScheduleHybrid());

    it('restrict index', () => testRestrictIndex());
    it('restrict active', () => testRestrictActive());

    it('prefer idle', () => testActivePriority());
    it('prefer lower load', () => testUtilPriority());

    it('zero gpu', () => testZeroGpu());
});

async function testInit(): Promise<void> {
    Helpers.mockGpuInfo(gpuInfo);
    scheduler = new TaskScheduler();
    await scheduler.init();
}

async function testScheduleSmall(): Promise<void> {
    const idx1 = await schedule('small', '1', 0.5);  // [0]
    const idx2 = await schedule('small', '2', 0.5);  // [1]
    const idx3 = await schedule('small', '3', 0.5);  // [2]
    const idx4 = await schedule('small', '4', 0.6);  // null
    const idx5 = await schedule('small', '5', 0.5);  // [0]

    assert.equal(idx4, null);

    const count = [ 0, 0, 0 ];  // count how many times each GPU is scheduled
    for (const idx of [ idx1, idx2, idx3, idx5 ]) {
        assert.notEqual(idx, null);
        assert.equal(idx!.length, 1);
        count[idx![0]] += 1;
    }
    assert.deepEqual(count.sort(), [ 1, 1, 2 ]);
}

async function testReleaseSmall(): Promise<void> {
    scheduler.releaseAll('small');
    const utils = Helpers.getGpuUtils(scheduler);
    assert.deepEqual(utils, [ 0, 0, 0 ]);
}

async function testScheduleLarge(): Promise<void> {
    const idx1 = await schedule('large1', 'x', 2);  // [0,1]
    const idx2 = await schedule('large2', 'x', 2);  // null
    const idx3 = await schedule('large3', 'x', 1);  // [2]

    assert.notEqual(idx1, null);
    assert.equal(idx1!.length, 2);

    assert.equal(idx2, null);

    assert.notEqual(idx3, null);
    assert.equal(idx3!.length, 1);

    assert.deepEqual([ ...idx1!, ...idx3! ].sort(), [ 0, 1, 2 ]);
}

async function testReleaseLarge(): Promise<void> {
    scheduler.release('large1', 'x');
    let utils = Helpers.getGpuUtils(scheduler);
    assert.deepEqual(utils.sort(), [ 0, 0, 1 ]);

    scheduler.release('large3', 'x');
    utils = Helpers.getGpuUtils(scheduler);
    assert.deepEqual(utils.sort(), [ 0, 0, 0 ]);
}

async function testScheduleHybrid(): Promise<void> {
    const idx1 = await schedule('small', '1', 0.5);  // [0]
    const idx2 = await schedule('large', '1', 1);  // [1]
    const idx3 = await schedule('large', '2', 2);  // null
    scheduler.release('large', '1');
    const idx4 = await schedule('large', '3', 2);  // [1,2]

    assert.notEqual(idx1, null);
    assert.equal(idx1!.length, 1);

    assert.notEqual(idx2, null);
    assert.equal(idx2!.length, 1);

    assert.equal(idx3, null);

    assert.notEqual(idx4, null);
    assert.equal(idx4!.length, 2);

    assert.notEqual(idx1![0], idx2![0]);
    assert.deepEqual([ ...idx1!, ...idx4! ].sort(), [ 0, 1, 2 ]);

    scheduler.releaseAll('small');
    scheduler.releaseAll('large');
}

async function testRestrictIndex(): Promise<void> {
    const idx1 = await schedule('r', '1', 0.5, { onlyUseIndices: [ 1 ] });
    const idx2 = await schedule('r', '2', 2, { onlyUseIndices: [ 1, 2 ] });
    const idx3 = await schedule('r', '3', 1, { onlyUseIndices: [ 1, 2 ] });

    assert.deepEqual(idx1, [ 1 ]);
    assert.equal(idx2, null);
    assert.deepEqual(idx3, [ 2 ]);

    scheduler.releaseAll('r');
}

async function testRestrictActive(): Promise<void> {
    gpuInfo.processes = [
        { gpuIndex: 0, type: 'graphics' },
        { gpuIndex: 1, type: 'compute' },
    ];
    await scheduler.update();

    const idx1 = await schedule('r', '1', 1, { rejectActive: true });
    const idx2 = await schedule('r', '2', 1, { rejectActive: true });
    const idx3 = await schedule('r', '3', 1, { rejectComputeActive: true });

    assert.deepEqual(idx1, [ 2 ]);
    assert.equal(idx2, null);
    assert.deepEqual(idx3, [ 0 ]);

    scheduler.releaseAll('r');
}

async function testActivePriority(): Promise<void> {
    gpuInfo.processes = [
        { gpuIndex: 0, type: 'graphics' },
        { gpuIndex: 1, type: 'compute' },
    ];
    await scheduler.update();

    const idx1 = await schedule('p', '1', 1);
    const idx2 = await schedule('p', '2', 1);
    const idx3 = await schedule('p', '3', 1);

    assert.deepEqual(idx1, [ 2 ]);
    assert.deepEqual(idx2, [ 0 ]);
    assert.deepEqual(idx3, [ 1 ]);

    scheduler.releaseAll('p');
}

async function testUtilPriority(): Promise<void> {
    gpuInfo.gpus[0].gpuCoreUtilization = 50;
    gpuInfo.gpus[1].gpuCoreUtilization = 10;
    gpuInfo.gpus[2].gpuMemoryUtilization = 20;
    gpuInfo.processes = [];
    await scheduler.update();

    const idx1 = await schedule('p', '1', 1);
    const idx2 = await schedule('p', '2', 1);
    const idx3 = await schedule('p', '3', 1);

    assert.deepEqual(idx1, [ 1 ]);
    assert.deepEqual(idx2, [ 0 ]);
    assert.deepEqual(idx3, [ 2 ]);

    scheduler.releaseAll('p');
}

async function testZeroGpu(): Promise<void> {
    const idx = await schedule('z', '1', 0);
    assert.deepEqual(idx, []);
}

async function schedule(expId: string, trialId: string, gpuNum: number, restrict?: TrialKeeper.GpuRestrictions):
        Promise<number[] | null> {

    const env = await scheduler.schedule(expId, trialId, gpuNum, restrict);
    if (env === null) {
        return null;
    }

    const indices = env['CUDA_VISIBLE_DEVICES'];
    if (indices === '') {
        return [];
    }

    return indices.split(',').map(Number);
}
