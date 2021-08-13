// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import * as chai from 'chai';
import { cleanupUnitTest, prepareUnitTest } from '../../../../common/utils';
import chaiAsPromised = require("chai-as-promised");
import { NFSSharedStorageService } from "../../../../training_service/reusable/shared_storages/nfsStorageService";
import { NfsConfig } from '../../../../common/experimentConfig';

describe('Unit Test for sharedstorage', () => {
    let service: NFSSharedStorageService;
    let testNFSConfig: NfsConfig = JSON.parse(
        "{\"storageType\": \"NFS\",\"localMountPoint\": \"localMountPoint\",\"remoteMountPoint\": \"remoteMountPoint\",\"localMounted\":\"localMounted\",\"nfsServer\":\"nfsServer\",\"exportedDirectory\":\"exportedDirectory\"}");

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
        service = new NFSSharedStorageService();
        service.config(testNFSConfig);
    });

    after(() => {
        cleanupUnitTest();
    });

    it('test nfsStorageService canLocalMounted', async () => {
        chai.expect(service.canLocalMounted).to.be.true;
    });

    it('test nfsStorageService localMountCommand', async () => {
        let serviceLocal: NFSSharedStorageService;
        serviceLocal = new NFSSharedStorageService();
        chai.expect(serviceLocal.localMountCommand).to.be.equal('');

        await serviceLocal.config(testNFSConfig)
        chai.expect(serviceLocal.localMountCommand).to.not.be.equal('');
    });

    it('test nfsStorageService storageService', async () => {
        chai.expect(service.storageService).to.not.be.equal(undefined);
    });

    it('test nfsStorageService remoteMountCommand', async () => {
        let serviceLocal: NFSSharedStorageService;
        serviceLocal = new NFSSharedStorageService();
        chai.expect(serviceLocal.remoteMountCommand).to.be.equal('');

        await serviceLocal.config(testNFSConfig)
        chai.expect(serviceLocal.remoteMountCommand).to.not.be.equal('');
    });

    it('test nfsStorageService remoteUmountCommand', async () => {
        let serviceLocal: NFSSharedStorageService;
        serviceLocal = new NFSSharedStorageService();
        chai.expect(serviceLocal.remoteUmountCommand).to.be.equal('');

        await serviceLocal.config(testNFSConfig)
        chai.expect(serviceLocal.remoteUmountCommand).to.not.be.equal('');
    });

    it('test nfsStorageService localWorkingRoot', async () => {
        chai.expect(service.localWorkingRoot).to.not.be.equal('');
    });

    it('test nfsStorageService remoteWorkingRoot', async () => {
        chai.expect(service.remoteWorkingRoot).to.not.be.equal('');
    });
});
