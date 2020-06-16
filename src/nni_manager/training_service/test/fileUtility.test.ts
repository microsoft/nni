// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';
import * as assert from 'assert';
import * as chai from 'chai';
import * as fs from 'fs';
import * as path from 'path';
import * as tar from 'tar';
import { execCopydir, tarAdd, validateCodeDir } from '../common/util';

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
        fs.writeFileSync(path.join(sourceDir, '.nniignore'), 'abc\nxyz');
        fs.writeFileSync(path.join(sourceDir, 'abc'), '123');
        fs.writeFileSync(path.join(sourceDir, 'abcd'), '1234');
        fs.mkdirSync(path.join(sourceDir, 'xyz'));
        fs.mkdirSync(path.join(sourceDir, 'xyy'));
        fs.mkdirSync(path.join(sourceDir, 'www'));
        fs.mkdirSync(path.join(sourceDir, 'xx'));  // empty dir
        fs.writeFileSync(path.join(sourceDir, 'xyy', '.nniignore'), 'qq');  // nested nniignore
        fs.writeFileSync(path.join(sourceDir, 'xyy', 'abc'), '123');
        fs.writeFileSync(path.join(sourceDir, 'xyy', 'qq'), '1234');
        fs.writeFileSync(path.join(sourceDir, 'xyy', 'pp'), '1234');
        fs.writeFileSync(path.join(sourceDir, 'www', '.nniignore'), 'pp');  // pop nniignore
        fs.writeFileSync(path.join(sourceDir, 'www', 'abc'), '123');
        fs.writeFileSync(path.join(sourceDir, 'www', 'qq'), '1234');
        fs.writeFileSync(path.join(sourceDir, 'www', 'pp'), '1234');
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
        const existFiles = [
            'abcd',
            'xyy',
            'xx',
            path.join('xyy', '.nniignore'),
            path.join('xyy', 'pp'),
            path.join('www', '.nniignore'),
            path.join('www', 'qq'),
        ]
        const notExistFiles = [
            'abc',
            'xyz',
            path.join('xyy', 'abc'),
            path.join('xyy', 'qq'),
            path.join('www', 'pp'),
            path.join('www', 'abc'),
        ]
        existFiles.forEach(d => assert.ok(fs.existsSync(path.join(destDir, d))));
        notExistFiles.forEach(d => assert.ok(!fs.existsSync(path.join(destDir, d))));
    });

    it('Test file copy without ignore', async () => {
        fs.unlinkSync(path.join(sourceDir, '.nniignore'));
        await execCopydir(sourceDir, destDir);
        assert.ok(fs.existsSync(path.join(destDir, 'abcd')));
        assert.ok(fs.existsSync(path.join(destDir, 'abc')));
        assert.ok(fs.existsSync(path.join(destDir, 'xyz')));
        assert.ok(fs.existsSync(path.join(destDir, 'xyy')));
        assert.ok(fs.existsSync(path.join(destDir, 'xx')));
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

    it('Validate code ok', async () => {
        assert.doesNotThrow(async () => validateCodeDir(sourceDir));
    });

    it('Validate code too many files', async () => {
        for (let i = 0; i < 2000; ++i)
            fs.writeFileSync(path.join(sourceDir, `${i}.txt`), 'a');
        try {
            await validateCodeDir(sourceDir);
        } catch (error) {
            chai.expect(error.message).to.contains('many files');
            return;
        }
        chai.expect.fail(null, null, 'Did not fail.');
    });

    it('Validate code too many files ok', async() => {
        for (let i = 0; i < 2000; ++i)
            fs.writeFileSync(path.join(sourceDir, `${i}.txt`), 'a');
        fs.writeFileSync(path.join(sourceDir, '.nniignore'), '*.txt');
        assert.doesNotThrow(async () => validateCodeDir(sourceDir));
    });
});
