// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as chai from 'chai';
import * as fs from 'fs';
import * as path from 'path';
import { getLogger, Logger } from "../../../common/log";
import { cleanupUnitTest, prepareUnitTest } from '../../../common/utils';
import { MountedStorageService } from "../storages/mountedStorageService";
import chaiAsPromised = require("chai-as-promised");


async function remove(removedPath: string, isDirectory: boolean, isRecursive: boolean): Promise<void> {
    if (isDirectory) {
        if (isRecursive) {
            const children = await fs.promises.readdir(removedPath);
            for (const fileName of children) {
                const filePath = path.join(removedPath, fileName);
                const stat = await fs.promises.lstat(filePath);
                await remove(filePath, stat.isDirectory(), isRecursive);
            }
        }
        await fs.promises.rmdir(removedPath);
    } else {
        await fs.promises.unlink(removedPath);
    }
}

describe('Unit Test for MountedStorageService', () => {

    let service: MountedStorageService;
    let log: Logger;
    let localPath = "reusableut/local";
    let mountedPath = "reusableut/mounted";

    const testPath = "testpath";
    const testFileName = "testfile.txt";
    let localCopiedPath: string;
    let localFileName: string;
    let mountedFileName: string;

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        prepareUnitTest();
        log = getLogger();

        const testRoot = path.dirname(__filename);
        localPath = path.join(testRoot, localPath);
        mountedPath = path.join(testRoot, mountedPath);
        service = new MountedStorageService();
        service.initialize(localPath, mountedPath);

        localCopiedPath = path.join(localPath, testPath);
        localFileName = path.join(localCopiedPath, testFileName);
        mountedFileName = path.join(testPath, testFileName);
    });

    after(() => {
        cleanupUnitTest();
    });

    beforeEach(async () => {
        if (!fs.existsSync(localPath)) {
            await fs.promises.mkdir(localPath, { recursive: true });
        }
        if (!fs.existsSync(mountedPath)) {
            await fs.promises.mkdir(mountedPath, { recursive: true });
        }
        log.info(`localFileName: ${localFileName}`);

        await fs.promises.mkdir(localCopiedPath, { recursive: true });
        await fs.promises.writeFile(localFileName, "hello world");
    });

    afterEach(async () => {
        const testRootPath = path.normalize(`${localPath}/../../reusableut`);
        await remove(testRootPath, true, true);
    });

    it('copyAndRename', async () => {
        await service.copyDirectory(localCopiedPath, ".");
        chai.expect(fs.existsSync(mountedPath));

        const newName = `${testFileName}new`;
        await service.rename(mountedFileName, newName);
        chai.assert.isFalse(fs.existsSync(testPath));
        const newTestPath = `${mountedFileName}new`;
        chai.assert.isTrue(await service.exists(newTestPath));

        await service.copyFileBack(newTestPath, ".");
        const localNewFileName = `${localPath}/${newName}`;
        chai.assert.isTrue(fs.existsSync(localNewFileName));

        fs.unlinkSync(`${localFileName}`);
        fs.rmdirSync(`${localPath}/${testPath}`);
        await service.copyDirectoryBack(`${mountedPath}/${testPath}`, `.`);
        const localNewName = `${localFileName}new`;
        chai.assert.isTrue(fs.existsSync(localNewName));
    })

    it('FileContentTest', async () => {
        const savedFileName = "savedfile.txt";
        await service.save("01234", savedFileName);
        chai.expect(fs.existsSync(savedFileName));

        let content = await service.readFileContent(savedFileName, 0, -1);
        chai.assert.equal(content, "01234");

        await service.save("56789", savedFileName, true);
        content = await service.readFileContent(savedFileName, 0, -1);
        chai.assert.equal(content, "0123456789");

        content = await service.readFileContent(savedFileName, -1, 1);
        chai.assert.equal(content, "0");

        content = await service.readFileContent(savedFileName, 5, 1);
        chai.assert.equal(content, "5");

        content = await service.readFileContent(savedFileName, 5, -1);
        chai.assert.equal(content, "56789");
    });
});
