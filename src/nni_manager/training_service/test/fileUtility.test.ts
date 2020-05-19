// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';
import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as tar from 'tar';
import { execCopydir, tarAdd } from '../common/util';

const deleteFolderRecursive = (filePath: string) => {
    if (fs.existsSync(filePath)) {
        fs.readdirSync(filePath).forEach((file, index) => {
            const curPath = path.join(filePath, file);
            if (fs.lstatSync(curPath).isDirectory()) { // recurse
                deleteFolderRecursive(curPath);
            } else { // delete file
                fs.unlinkSync(curPath);
            }
        });
        fs.rmdirSync(filePath);
    }
};

describe('fileUtility', () => {
    /*
    Test file utilities, includes:
    - Copy directory
    - Ignore with ignore file
    - Add to tar
    */

    const sourceDir = 'test-fileUtilityTestSource';
    const destDir = 'test-fileUtilityTestDest';

    beforeEach(() => {
        fs.mkdirSync(sourceDir);
        fs.writeFileSync(path.join(sourceDir, '.nniignore'), 'abc');
        fs.writeFileSync(path.join(sourceDir, 'abc'), '123');
        fs.writeFileSync(path.join(sourceDir, 'abcd'), '1234');
    });

    afterEach(() => {
        deleteFolderRecursive(sourceDir);
        deleteFolderRecursive(destDir);
        if (fs.existsSync(`${destDir}.tar`)) {
            fs.unlinkSync(`${destDir}.tar`);
        }
    });

    it('Test file copy', async () => {
        await execCopydir(sourceDir, destDir);
        assert.ok(fs.existsSync(path.join(destDir, 'abcd')));
        assert.ok(!fs.existsSync(path.join(destDir, 'abc')));
    });

    it('Test file copy without ignore', async () => {
        fs.unlinkSync(path.join(sourceDir, '.nniignore'));
        await execCopydir(sourceDir, destDir);
        assert.ok(fs.existsSync(path.join(destDir, 'abcd')));
        assert.ok(fs.existsSync(path.join(destDir, 'abc')));
    });

    it('Test tar file', async () => {
        const tarPath = `${destDir}.tar`;
        await tarAdd(tarPath, sourceDir);
        assert.ok(fs.existsSync(tarPath));
        fs.mkdirSync(destDir);
        tar.extract({
            file: tarPath,
            cwd: destDir,
            sync: true
        })
        assert.ok(fs.existsSync(path.join(destDir, 'abcd')));
        assert.ok(!fs.existsSync(path.join(destDir, 'abc')));
    });
});
