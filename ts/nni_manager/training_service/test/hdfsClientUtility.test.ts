// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';
import * as chai from 'chai';
import * as chaiAsPromised from 'chai-as-promised';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as tmp from 'tmp';
import { cleanupUnitTest, prepareUnitTest, uniqueString } from '../../common/utils';
import { HDFSClientUtility } from '../pai/paiYarn/hdfsClientUtility';

var WebHDFS = require('webhdfs');
var rmdir = require('rmdir');

describe('WebHDFS', function () {
    /*
    To enable web HDFS client unit test, HDFS information needs to be configured in:
    Default/.vscode/hdfsInfo.json,  whose content looks like:
    {
        "user": "user1",
        "port": 50070,
        "host": "10.0.0.0"
    }
    */
    let skip: boolean = false;
    let testHDFSInfo: any;
    let hdfsClient: any;
    try {
        testHDFSInfo = JSON.parse(fs.readFileSync('../../.vscode/hdfsInfo.json', 'utf8'));
        console.log(testHDFSInfo);
        hdfsClient = WebHDFS.createClient({
            user: testHDFSInfo.user,
            port: testHDFSInfo.port,
            host: testHDFSInfo.host
        });
    } catch (err) {
        console.log('Please configure rminfo.json to enable remote machine unit test.');
        skip = true;
    }

    before(() => {
        chai.should();
        chai.use(chaiAsPromised);
        tmp.setGracefulCleanup();
        prepareUnitTest();
    });

    after(() => {
        cleanupUnitTest();
    });

    it('Test HDFS utility path functions', async () => {
        if (skip) {
            return;
        }
        const testPath : string = '/nni_unittest_' + uniqueString(6);
        let exists : boolean = await HDFSClientUtility.pathExists(testPath, hdfsClient);
        // The new random named path is expected to not exist
        chai.expect(exists).to.be.equals(false);

        const mkdirResult : boolean = await HDFSClientUtility.mkdir(testPath, hdfsClient);
        // Mkdir is expected to be successful
        chai.expect(mkdirResult).to.be.equals(true);

        exists = await HDFSClientUtility.pathExists(testPath, hdfsClient);
        // The newly created path is expected to exist
        chai.expect(exists).to.be.equals(true);

        const deleteResult : boolean = await HDFSClientUtility.deletePath(testPath, hdfsClient);
        // Delete path is expected to be successful
        chai.expect(deleteResult).to.be.equals(true);

        exists = await HDFSClientUtility.pathExists(testPath, hdfsClient);
        // The deleted path is not expected to exist
        chai.expect(exists).to.be.equals(false);
    });

    it('Test HDFS utility copyFileToHdfs', async() => {
        if (skip) {
            return;
        }
        // Prepare local directory and files
        const tmpLocalDirectoryPath : string = path.join(os.tmpdir(), 'nni_unittest_dir_' + uniqueString(6));
        const tmpDataFilePath : string = path.join(tmpLocalDirectoryPath, 'file_' + uniqueString(6));
        const testFileData : string = 'TestContent123';
        fs.mkdirSync(tmpLocalDirectoryPath);
        fs.writeFileSync(tmpDataFilePath, testFileData);

        const testHDFSFilePath : string = '/nni_unittest_' + uniqueString(6);
        let exists : boolean = await HDFSClientUtility.pathExists(testHDFSFilePath, hdfsClient);
        // The new random named path is expected to not exist
        chai.expect(exists).to.be.equals(false);

        await HDFSClientUtility.copyFileToHdfs(tmpDataFilePath, testHDFSFilePath, hdfsClient);
        exists = await HDFSClientUtility.pathExists(testHDFSFilePath, hdfsClient);
        // After copy local file to HDFS, the target file path in HDFS is expected to exist
        chai.expect(exists).to.be.equals(true);

        const buffer : Buffer = await HDFSClientUtility.readFileFromHDFS(testHDFSFilePath, hdfsClient);
        const actualFileData : string = buffer.toString('utf8');
        // The file content read from HDFS is expected to equal to the content of local file
        chai.expect(actualFileData).to.be.equals(testFileData);

        const testHDFSDirPath : string = path.join('/nni_unittest_' + uniqueString(6) +  '_dir');

        await HDFSClientUtility.copyDirectoryToHdfs(tmpLocalDirectoryPath, testHDFSDirPath, hdfsClient);

        const files : any[] = await HDFSClientUtility.readdir(testHDFSDirPath, hdfsClient);

        // Expected file count under HDFS target directory is 1
        chai.expect(files.length).to.be.equals(1);

        // Expected file name under HDFS target directory is equal to local file name
        chai.expect(files[0].pathSuffix).to.be.equals(path.parse(tmpDataFilePath).base);

        // Cleanup
        rmdir(tmpLocalDirectoryPath);

        let deleteRestult : boolean = await HDFSClientUtility.deletePath(testHDFSFilePath, hdfsClient);
        chai.expect(deleteRestult).to.be.equals(true);

        deleteRestult = await HDFSClientUtility.deletePath(testHDFSDirPath, hdfsClient);
        chai.expect(deleteRestult).to.be.equals(true);
    });
});
