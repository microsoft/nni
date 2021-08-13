// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import * as chai from 'chai';
import { cleanupUnitTest, prepareUnitTest } from '../../../../common/utils';
import chaiAsPromised = require("chai-as-promised");
import { AzureBlobSharedStorageService } from "../../../../training_service/reusable/shared_storages/azureblobStorageService";
import { AzureBlobConfig } from '../../../../common/experimentConfig';

describe('Unit Test for AzureBlobSharedstorage', () => {
    let service: AzureBlobSharedStorageService;
    let testAzureBlobConfig: AzureBlobConfig = JSON.parse(
        "{\"storageType\": \"AzureBlob\",\"localMountPoint\": \"localMountPoint\",\"remoteMountPoint\": \"remoteMountPoint\",\"localMounted\":\"localMounted\",\"storageAccountName\":\"storageAccountName\",\"storageAccountKey\":\"storageAccountKey\",\"containerName\":\"containerName\"}");

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
        service = new AzureBlobSharedStorageService();
        service.config(testAzureBlobConfig);
    });

    after(() => {
        cleanupUnitTest();
    });

    it('test azureblobStorageService canLocalMounted', async () => {
        chai.expect(service.canLocalMounted).to.be.true;
    });

    it('test azureblobStorageService localMountCommand', async () => {
        let serviceLocal: AzureBlobSharedStorageService;
        serviceLocal = new AzureBlobSharedStorageService();
        chai.expect(serviceLocal.localMountCommand).to.be.equal('');

        await serviceLocal.config(testAzureBlobConfig)
        chai.expect(serviceLocal.localMountCommand).to.not.be.equal('');
    });

    it('test azureblobStorageService storageService', async () => {
        chai.expect(service.storageService).to.not.be.equal(undefined);
    });

    it('test azureblobStorageService remoteMountCommand', async () => {
        let serviceLocal: AzureBlobSharedStorageService;
        serviceLocal = new AzureBlobSharedStorageService();
        chai.expect(serviceLocal.remoteMountCommand).to.be.equal('');

        await serviceLocal.config(testAzureBlobConfig)
        chai.expect(serviceLocal.remoteMountCommand).to.not.be.equal('');
    });

    it('test azureblobStorageService remoteUmountCommand', async () => {
        let serviceLocal: AzureBlobSharedStorageService;
        serviceLocal = new AzureBlobSharedStorageService();
        chai.expect(serviceLocal.remoteUmountCommand).to.be.equal('');

        await serviceLocal.config(testAzureBlobConfig)
        chai.expect(serviceLocal.remoteUmountCommand).to.not.be.equal('');
    });

    it('test azureblobStorageService localWorkingRoot', async () => {
        chai.expect(service.localWorkingRoot).to.not.be.equal('');
    });

    it('test azureblobStorageService remoteWorkingRoot', async () => {
        chai.expect(service.remoteWorkingRoot).to.not.be.equal('');
    });
});
