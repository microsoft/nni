// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import * as chai from 'chai';
import * as path from 'path';
import { Scope } from "typescript-ioc";
import * as component from '../../../common/component';
import { getLogger, Logger } from "../../../common/log";
import { TrialJobApplicationForm, TrialJobStatus } from '../../../common/trainingService';
import { cleanupUnitTest, delay, prepareUnitTest } from '../../../common/utils';
import { INITIALIZED, KILL_TRIAL_JOB, NEW_TRIAL_JOB, SEND_TRIAL_JOB_PARAMETER, TRIAL_END } from '../../../core/commands';
import { TrialConfigMetadataKey } from '../../../training_service/common/trialConfigMetadataKey';
import { Command } from '../commandChannel';
import { EnvironmentInformation, EnvironmentService } from "../environment";
import { TrialDetail } from '../trial';
import { TrialDispatcher } from "../trialDispatcher";
import { UtCommandChannel } from './utCommandChannel';
import { UtEnvironmentService } from "./utEnvironmentService";
import chaiAsPromised = require("chai-as-promised");

function createTrialForm(content: any = undefined): TrialJobApplicationForm {
    if (content === undefined) {
        content = {
            "test": 1
        };
    }
    const trialForm = {
        sequenceId: 0,
        hyperParameters: {
            value: JSON.stringify(content),
            index: 0
        }
    };
    return trialForm;
}

async function waitResult<TResult>(callback: () => Promise<TResult | undefined>, waitMs: number = 1000, interval: number = 1, throwError: boolean = false): Promise<TResult | undefined> {
    while (waitMs > 0) {
        const result = await callback();
        if (result !== undefined) {
            return result;
        }
        await delay(interval);
        waitMs -= interval;
    };

    if (throwError) {
        throw new Error(`wait result timeout!\n${callback.toString()}`);
    }

    return undefined;
}

async function waitResultMust<TResult>(callback: () => Promise<TResult | undefined>, waitMs: number = 1000, interval: number = 1): Promise<TResult> {
    const result = await waitResult(callback, waitMs, interval, true);
    // this error should be thrown in waitResult already.
    if (result === undefined) {
        throw new Error(`wait result timeout!`);
    }
    return result;
}

async function newTrial(trialDispatcher: TrialDispatcher): Promise<TrialDetail> {
    const trialDetail = await trialDispatcher.submitTrialJob(createTrialForm());

    return trialDetail;
}

async function verifyTrialRunning(commandChannel: UtCommandChannel, trialDetail: TrialDetail): Promise<void> {

    let command = await waitResultMust<Command>(async () => {
        return await commandChannel.testReceiveCommandFromTrialDispatcher();
    });
    chai.assert.equal(command.command, NEW_TRIAL_JOB);
    chai.assert.equal(command.data["trialId"], trialDetail.id);
}

async function verifyTrialResult(commandChannel: UtCommandChannel, trialDetail: TrialDetail, returnCode: number = 0): Promise<void> {
    let trialResult = {
        trial: trialDetail.id,
        code: returnCode,
        timestamp: Date.now(),
    };
    if (trialDetail.environment === undefined) {
        throw new Error(`environment shouldn't be undefined.`)
    }

    await commandChannel.testSendCommandToTrialDispatcher(trialDetail.environment, TRIAL_END, trialResult);
    await waitResultMust<boolean>(async () => {
        return trialDetail.status !== 'RUNNING' ? true : undefined;
    });
    if (returnCode === 0) {
        chai.assert.equal<TrialJobStatus>(trialDetail.status, 'SUCCEEDED', "trial should be succeeded");
    } else {
        chai.assert.equal<TrialJobStatus>(trialDetail.status, 'FAILED', "trial should be failed");
    }
}

async function waitEnvironment(waitCount: number, previousEnvironments: Map<string, EnvironmentInformation>, environmentService: UtEnvironmentService, commandChannel: UtCommandChannel): Promise<EnvironmentInformation> {
    const initalizedMessage = {
        nodeId: null,
    }
    const waitRequestEnvironment = await waitResultMust<EnvironmentInformation>(async () => {
        const environments = environmentService.testGetEnvironments();
        if (environments.size === waitCount) {
            for (const [id, environment] of environments) {
                if (!previousEnvironments.has(id)) {
                    previousEnvironments.set(id, environment);
                    return environment;
                }
            }
        }
        return undefined;
    });

    if (waitRequestEnvironment === undefined) {
        throw new Error(`waitRequestEnvironment is not defined.`);
    }
    // set env to running
    environmentService.testSetEnvironmentStatus(waitRequestEnvironment, 'RUNNING');
    // set runner is ready.
    await commandChannel.testSendCommandToTrialDispatcher(waitRequestEnvironment, INITIALIZED, initalizedMessage);
    return waitRequestEnvironment;
}

describe('Unit Test for TrialDispatcher', () => {

    let trialRunPromise: Promise<void>;
    let trialDispatcher: TrialDispatcher;
    let commandChannel: UtCommandChannel;
    let environmentService: UtEnvironmentService;
    let log: Logger;
    let previousEnvironments: Map<string, EnvironmentInformation> = new Map<string, EnvironmentInformation>();

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
        log = getLogger();
    });

    after(() => {
        cleanupUnitTest();
    });

    beforeEach(async () => {
        const currentDir = path.dirname(__filename);
        const trialConfig = {
            codeDir: currentDir,
        }
        const nniManagerIpConfig = {
            nniManagerIp: "127.0.0.1",
        }
        trialDispatcher = new TrialDispatcher();
        component.Container.bind(EnvironmentService)
            .to(UtEnvironmentService)
            .scope(Scope.Singleton);

        await trialDispatcher.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, JSON.stringify(trialConfig));
        await trialDispatcher.setClusterMetadata(TrialConfigMetadataKey.NNI_MANAGER_IP, JSON.stringify(nniManagerIpConfig));
        trialRunPromise = trialDispatcher.run();

        environmentService = component.get(EnvironmentService) as UtEnvironmentService;
        commandChannel = environmentService.testGetCommandChannel();
    });

    afterEach(async () => {
        previousEnvironments.clear();
        await trialDispatcher.cleanUp();
        environmentService.testReset();
        await trialRunPromise;
    });

    it('reuse env', async () => {
        // submit first trial
        let trialDetail = await newTrial(trialDispatcher);
        // wait env started
        await waitEnvironment(1, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail);
        await verifyTrialResult(commandChannel, trialDetail, 0);

        trialDetail = await newTrial(trialDispatcher);
        await verifyTrialRunning(commandChannel, trialDetail);
        await verifyTrialResult(commandChannel, trialDetail, -1);

        chai.assert.equal(environmentService.testGetEnvironments().size, 1, "as env reused, so only 1 env should be here.");
        const trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 2, "there should be 2 trials");
    });

    it('not reusable env', async () => {
        trialDispatcher.setClusterMetadata(
            TrialConfigMetadataKey.TRIAL_CONFIG,
            JSON.stringify({
                reuseEnvironment: false,
                codeDir: path.dirname(__filename),
            }));
        // submit first trial
        let trialDetail = await newTrial(trialDispatcher);
        // wait env started
        let environment = await waitEnvironment(1, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail);
        await verifyTrialResult(commandChannel, trialDetail, 0);
        await waitResultMust<true>(async () => {
            return environment.status === 'USER_CANCELED' ? true : undefined;
        });

        trialDetail = await newTrial(trialDispatcher);
        // wait env started
        await waitEnvironment(2, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail);
        await verifyTrialResult(commandChannel, trialDetail, -1);
        await waitResultMust<true>(async () => {
            return environment.status === 'USER_CANCELED' ? true : undefined;
        });

        chai.assert.equal(environmentService.testGetEnvironments().size, 2, "as env not reused, so only 2 envs should be here.");
        const trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 2, "there should be 2 trials");
    });

    it('no more env', async () => {
        // submit first trial
        const trialDetail1 = await newTrial(trialDispatcher);
        // wait env started
        await waitEnvironment(1, previousEnvironments, environmentService, commandChannel);

        // set to no more environment
        environmentService.testSetNoMoreEnvironment(false);

        const trialDetail2 = await newTrial(trialDispatcher);

        await verifyTrialRunning(commandChannel, trialDetail1);
        await verifyTrialResult(commandChannel, trialDetail1, 0);

        // wait env started
        await verifyTrialRunning(commandChannel, trialDetail2);
        await verifyTrialResult(commandChannel, trialDetail2, -1);

        chai.assert.equal(environmentService.testGetEnvironments().size, 1, "as env not reused, so only 1 envs should be here.");
        const trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 2, "there should be 2 trials");
    });


    it('2trial2env', async () => {
        // submit first trial
        let trialDetail1 = await newTrial(trialDispatcher);
        let trialDetail2 = await newTrial(trialDispatcher);

        // wait env started
        await waitEnvironment(2, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail1);
        await verifyTrialResult(commandChannel, trialDetail1, 0);
        await verifyTrialRunning(commandChannel, trialDetail2);
        await verifyTrialResult(commandChannel, trialDetail2, 0);

        chai.assert.equal(environmentService.testGetEnvironments().size, 2, "2 envs should be here.");
        const trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 2, "there should be 2 trials");
    });

    it('3trial2env', async () => {
        // submit first trial
        let trialDetail1 = await newTrial(trialDispatcher);
        let trialDetail2 = await newTrial(trialDispatcher);

        // wait env started
        await waitEnvironment(2, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail1);
        await verifyTrialResult(commandChannel, trialDetail1, 0);
        await verifyTrialRunning(commandChannel, trialDetail2);
        await verifyTrialResult(commandChannel, trialDetail2, 0);

        chai.assert.equal(environmentService.testGetEnvironments().size, 2, "2 envs should be here.");
        let trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 2, "there should be 2 trials");


        let trialDetail3 = await newTrial(trialDispatcher);
        await verifyTrialRunning(commandChannel, trialDetail3);
        await verifyTrialResult(commandChannel, trialDetail3, 0);

        chai.assert.equal(environmentService.testGetEnvironments().size, 2, "2 envs should be here.");
        trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 3, "there should be 2 trials");
    });

    it('stop trial', async () => {
        // submit first trial
        let trialDetail1 = await newTrial(trialDispatcher);
        // wait env started
        await waitEnvironment(1, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail1);
        await trialDispatcher.cancelTrialJob(trialDetail1.id, false);

        let command = await waitResultMust<Command>(async () => {
            return await commandChannel.testReceiveCommandFromTrialDispatcher();
        });
        chai.assert.equal(command.command, KILL_TRIAL_JOB);
        log.info(`command: ${JSON.stringify(command)}`);
        chai.assert.equal(command.data, trialDetail1.id);

        await waitResultMust<boolean>(async () => {
            return trialDetail1.status !== 'RUNNING' ? true : undefined;
        });

        let trialDetail2 = await newTrial(trialDispatcher);
        await verifyTrialRunning(commandChannel, trialDetail2);
        await trialDispatcher.cancelTrialJob(trialDetail2.id, true);
        command = await waitResultMust<Command>(async () => {
            return await commandChannel.testReceiveCommandFromTrialDispatcher();
        });
        chai.assert.equal(command.command, KILL_TRIAL_JOB);
        log.info(`command: ${JSON.stringify(command)}`);
        chai.assert.equal(command.data, trialDetail2.id);
        await waitResultMust<boolean>(async () => {
            return trialDetail2.status !== 'RUNNING' ? true : undefined;
        });

        chai.assert.equal(environmentService.testGetEnvironments().size, 1, "only one trial, so one env");
        const trials = await trialDispatcher.listTrialJobs();

        chai.assert.equal(trials.length, 2, "there should be 1 stopped trial only");
        let trial = await trialDispatcher.getTrialJob(trialDetail1.id);
        chai.assert.equal<TrialJobStatus>(trial.status, 'USER_CANCELED', `trial is canceled.`);
        trial = await trialDispatcher.getTrialJob(trialDetail2.id);
        chai.assert.equal<TrialJobStatus>(trial.status, 'EARLY_STOPPED', `trial is earlier stopped.`);
    });

    it('multi phase', async () => {
        let trialDetail = await newTrial(trialDispatcher);

        await waitEnvironment(1, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail);

        let content = {
            test: 2,
        }
        await trialDispatcher.updateTrialJob(trialDetail.id, createTrialForm(content));

        let command = await waitResultMust<Command>(async () => {
            return await commandChannel.testReceiveCommandFromTrialDispatcher();
        });

        chai.assert.equal(command.command, SEND_TRIAL_JOB_PARAMETER);
        chai.assert.equal(command.data["trialId"], trialDetail.id);
        chai.assert.equal(command.data.parameters.index, 0);
        chai.assert.equal(command.data.parameters.value, JSON.stringify(content));

        content = {
            test: 3,
        }
        await trialDispatcher.updateTrialJob(trialDetail.id, createTrialForm(content));
        command = await waitResultMust<Command>(async () => {
            return await commandChannel.testReceiveCommandFromTrialDispatcher();
        });
        chai.assert.equal(command.command, SEND_TRIAL_JOB_PARAMETER);
        chai.assert.equal(command.data["trialId"], trialDetail.id);
        chai.assert.equal(command.data.parameters.index, 0);
        chai.assert.equal(command.data.parameters.value, JSON.stringify(content));

        await verifyTrialResult(commandChannel, trialDetail, 0);

        chai.assert.equal(environmentService.testGetEnvironments().size, 1, "only one trial, so one env");
        const trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 1, "there should be 1 stopped trial only");
    });

    it('multi node', async () => {
        let trialDetail = await newTrial(trialDispatcher);

        const environment = await waitEnvironment(1, previousEnvironments, environmentService, commandChannel);
        environment.nodeCount = 2;
        await verifyTrialRunning(commandChannel, trialDetail);
        await verifyTrialResult(commandChannel, trialDetail, 0);

        let command = await waitResultMust<Command>(async () => {
            return await commandChannel.testReceiveCommandFromTrialDispatcher();
        });
        chai.assert.equal(command.command, KILL_TRIAL_JOB);
        chai.assert.equal(environmentService.testGetEnvironments().size, 1, "only one trial, so one env");
        const trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 1, "there should be 1 stopped trial only");
    });

    it('env timeout', async () => {
        let trialDetail = await newTrial(trialDispatcher);
        let environment = await waitEnvironment(1, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail);
        await verifyTrialResult(commandChannel, trialDetail, 0);

        environmentService.testSetEnvironmentStatus(environment, 'SUCCEEDED');
        await waitResultMust<boolean>(async () => {
            return environment.status === 'SUCCEEDED' ? true : undefined;
        });

        trialDetail = await newTrial(trialDispatcher);
        // wait env started
        await waitEnvironment(2, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail);
        await verifyTrialResult(commandChannel, trialDetail, 0);

        chai.assert.equal(previousEnvironments.size, 2, "as an env timeout, so 2 envs should be here.");
        const trials = await trialDispatcher.listTrialJobs();
        chai.assert.equal(trials.length, 2, "there should be 2 trials");
    });

    it('env failed with trial', async () => {
        let trialDetail = await newTrial(trialDispatcher);
        let environment = await waitEnvironment(1, previousEnvironments, environmentService, commandChannel);
        await verifyTrialRunning(commandChannel, trialDetail);

        environmentService.testSetEnvironmentStatus(environment, 'FAILED');
        await waitResultMust<boolean>(async () => {
            return environment.status === 'FAILED' ? true : undefined;
        });

        await waitResultMust<boolean>(async () => {
            return trialDetail.status === 'FAILED' ? true : undefined;
        });

        chai.assert.equal<TrialJobStatus>(trialDetail.status, 'FAILED', "env failed, so trial also failed.");
    });

    it('GPUScheduler disabled gpuNum === undefined', async () => {
        chai.assert.fail(`not implemented.`)
    });

    it('GPUScheduler disabled gpuNum === 0', async () => {
        chai.assert.fail(`not implemented.`)
    });

    it('GPUScheduler disabled multi node', async () => {
        chai.assert.fail(`not implemented.`)
    });

    it('GPUScheduler enabled 2 gpus 2 trial', async () => {
        chai.assert.fail(`not implemented.`)
    });

    it('GPUScheduler enabled use active gpus', async () => {
        chai.assert.fail(`not implemented.`)
    });

    it('GPUScheduler enabled TMP_NO_AVAILABLE_GPU, but with waiting env', async () => {
        chai.assert.fail(`not implemented.`)
    });

    it('GPUScheduler enabled TMP_NO_AVAILABLE_GPU, need env', async () => {
        chai.assert.fail(`not implemented.`)
    });

    it('GPUScheduler enabled REQUIRE_EXCEED_TOTAL, need fail', async () => {
        chai.assert.fail(`not implemented.`)
    });

    it('GPUScheduler enabled 2 trials on same gpu, 4 trials, 2 gpus', async () => {
        chai.assert.fail(`not implemented.`)
    });
});
