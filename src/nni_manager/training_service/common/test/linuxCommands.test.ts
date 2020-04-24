// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';
import * as fs from 'fs';
import * as component from '../../../common/component';
import { cleanupUnitTest, prepareUnitTest } from '../../../common/utils';
import { LinuxCommands } from '../extends/linuxCommands';
// import { TrialConfigMetadataKey } from '../trialConfigMetadataKey';


describe('Unit Test for linuxCommands', () => {
    /*
    To enable remote machine unit test, remote machine information needs to be configured in:
    Default/.vscode/rminfo.json,  whose content looks like:
    {
        "ip": "10.172.121.40",
        "user": "user1",
        "password": "mypassword"
    }
    */
    // let skip: boolean = false;
    // let testRmInfo: any;
    // let machineList: any;
    // try {
    //     testRmInfo = JSON.parse(fs.readFileSync('../../.vscode/rminfo.json', 'utf8'));
    //     console.log(testRmInfo);
    //     machineList = `[{\"ip\":\"${testRmInfo.ip}\",\"port\":22,\"username\":\"${testRmInfo.user}\",\"passwd\":\"${testRmInfo.password}\"}]`;
    // } catch (err) {
    //     console.log('Please configure rminfo.json to enable remote machine unit test.');
    //     skip = true;
    // }

    let linuxCommands: LinuxCommands

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
    });

    beforeEach(() => {
        linuxCommands = component.get(LinuxCommands);
        // remoteMachineTrainingService.run();
    });

    afterEach(() => {
        // remoteMachineTrainingService.cleanUp();
    });

    it('join path', async () => {
        chai.expect(linuxCommands.joinPath("/root/", "/first")).to.equal("/root/first")
        chai.expect(linuxCommands.joinPath("/root", "first")).to.equal("/root/first")
        chai.expect(linuxCommands.joinPath("/root/", "first")).to.equal("/root/first")
        chai.expect(linuxCommands.joinPath("root/", "first")).to.equal("root/first")
        chai.expect(linuxCommands.joinPath("root/")).to.equal("root/")
        chai.expect(linuxCommands.joinPath("root")).to.equal("root")
        chai.expect(linuxCommands.joinPath("./root")).to.equal("./root")
        chai.expect(linuxCommands.joinPath("")).to.equal(".")
        chai.expect(linuxCommands.joinPath("..")).to.equal("..")
    })

    it('remove folder', async () => {
        chai.expect(linuxCommands.removeFolder("test")).to.equal("rm -df 'test'")
        chai.expect(linuxCommands.removeFolder("test", true)).to.equal("rm -rf 'test'");
        chai.expect(linuxCommands.removeFolder("test", true, false)).to.equal("rm -r 'test'");
        chai.expect(linuxCommands.removeFolder("test", false, false)).to.equal("rm 'test'");
    })

    it('create folder', async () => {
        chai.expect(linuxCommands.createFolder("test")).to.equal("mkdir -p 'test'")
        chai.expect(linuxCommands.createFolder("test", true)).to.equal("umask 0; mkdir -p 'test'")
    })

    // it('List trial jobs', async () => {
    //     if (skip) {
    //         return;
    //     }
    //     chai.expect(await remoteMachineTrainingService.listTrialJobs()).to.be.empty;
    // });

    // it('Set cluster metadata', async () => {
    //     if (skip) {
    //         return;
    //     }
    //     await remoteMachineTrainingService.setClusterMetadata(TrialConfigMetadataKey.MACHINE_LIST, machineList);
    //     await remoteMachineTrainingService.setClusterMetadata(
    //         TrialConfigMetadataKey.TRIAL_CONFIG, `{"command":"sleep 1h && echo ","codeDir":"${localCodeDir}","gpuNum":1}`);
    //     const form: TrialJobApplicationForm = {
    //         sequenceId: 0,
    //         hyperParameters: {
    //             value: 'mock hyperparameters',
    //             index: 0
    //         }
    //     };
    //     const trialJob = await remoteMachineTrainingService.submitTrialJob(form);

    //     // After a job is cancelled, the status should be changed to 'USER_CANCELED'
    //     await remoteMachineTrainingService.cancelTrialJob(trialJob.id);

    //     // After a job is cancelled, the status should be changed to 'USER_CANCELED'
    //     const trialJob2 = await remoteMachineTrainingService.getTrialJob(trialJob.id);
    //     chai.expect(trialJob2.status).to.be.equals('USER_CANCELED');

    //     //Expect rejected if passing invalid trial job id
    //     await remoteMachineTrainingService.cancelTrialJob(trialJob.id + 'ddd').should.eventually.be.rejected;
    // });

    // it('Submit job test', async () => {
    //     if (skip) {
    //         return;
    //     }
    // });

    // it('Submit job and read metrics data', async () => {
    //     if (skip) {
    //         return;
    //     }
    //     // set machine list'
    //     await remoteMachineTrainingService.setClusterMetadata(TrialConfigMetadataKey.MACHINE_LIST, machineList);

    //     // set meta data
    //     const trialConfig: string = `{\"command\":\"python3 mockedTrial.py\", \"codeDir\":\"${localCodeDir}\",\"gpuNum\":0}`
    //     await remoteMachineTrainingService.setClusterMetadata(TrialConfigMetadataKey.TRIAL_CONFIG, trialConfig);

    //     // submit job
    //     const form: TrialJobApplicationForm = {
    //         sequenceId: 0,
    //         hyperParameters: {
    //             value: 'mock hyperparameters',
    //             index: 0
    //         }
    //     };
    //     const jobDetail: TrialJobDetail = await remoteMachineTrainingService.submitTrialJob(form);
    //     // Add metrics listeners
    //     const listener1 = function f1(metric: any) {
    //     }

    //     const listener2 = function f1(metric: any) {
    //     }

    //     remoteMachineTrainingService.addTrialJobMetricListener(listener1);
    //     remoteMachineTrainingService.addTrialJobMetricListener(listener2);
    //     await delay(10000);
    //     // remove listender1
    //     remoteMachineTrainingService.removeTrialJobMetricListener(listener1);
    //     await delay(5000);
    // }).timeout(30000);

    // it('Test getTrialJob exception', async () => {
    //     if (skip) {
    //         return;
    //     }
    //     await remoteMachineTrainingService.getTrialJob('wrongid').catch((err) => {
    //         assert(err !== undefined);
    //     });
    // });
});
