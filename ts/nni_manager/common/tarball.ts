// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import cpp from 'child-process-promise';
import cp from 'child_process';
import fs from 'fs';
import ignore from 'ignore';
import path from 'path';
import tar from 'tar';
import { getLogger } from 'common/log';
import { String } from 'typescript-string-operations';
import { GPU_INFO_COLLECTOR_FORMAT_WINDOWS } from './gpuData';

export async function createTarball(tarName: string, sourcePath: string): Promise<string> {
    const fileList = [];
    for (const d of listDirWithIgnoredFiles(sourcePath, '', [])) {
        fileList.push(d);
    }
    tar.create(
        {
            gzip: true,
            file: tarPath,
            sync: true,
            cwd: sourcePath,
        },
        fileList
    );
    return Promise.resolve();
}

class Walker {
    constructor() {
    }

    public async walk(): Promise<string> {
        const files = await fs.readdir(this.path);
    }
}

export function* listDirWithIgnoredFiles(root: string, relDir: string, ignoreFiles: string[]): Iterable<string> {
    let ignoreFile = undefined;
    const source = path.join(root, relDir);
    if (fs.existsSync(path.join(source, '.nniignore'))) {
        ignoreFile = path.join(source, '.nniignore');
        ignoreFiles.push(ignoreFile);
    }
    const ig = ignore();
    ignoreFiles.forEach((i) => ig.add(fs.readFileSync(i).toString()));
    for (const d of fs.readdirSync(source)) {
        const entry = path.join(relDir, d);
        if (ig.ignores(entry))
            continue;
        const entryStat = fs.statSync(path.join(root, entry));
        if (entryStat.isDirectory()) {
            yield entry;
            yield* listDirWithIgnoredFiles(root, entry, ignoreFiles);
        }
        else if (entryStat.isFile())
            yield entry;
    }
    if (ignoreFile !== undefined) {
        ignoreFiles.pop();
    }
}

export async function validateCodeDir(codeDir: string): Promise<number> {
    let fileCount: number = 0;
    let fileTotalSize: number = 0;
    for (const relPath of listDirWithIgnoredFiles(codeDir, '', [])) {
        const d = path.join(codeDir, relPath);
        fileCount += 1;
        fileTotalSize += fs.statSync(d).size;
        if (fileCount > 2000) {
            throw new Error(`Too many files and directories (${fileCount} already scanned) in ${codeDir},`
                + ` please check if it's a valid code dir`);
        }
        if (fileTotalSize > 300 * 1024 * 1024) {
            throw new Error(`File total size too large in code dir (${fileTotalSize} bytes already scanned, exceeds 300MB).`);
        }
        // NOTE: We added this test in case any training service or shared storage (e.g. HDFS) does not support complex file name.
        // If there is no bug found for long time, feel free to remove it.
        const fileNameValid = relPath.split(path.sep).every(fpart => (fpart.match('^[a-z0-9A-Z._-]*$') !== null));
        if (!fileNameValid) {
            const message = [
                `File ${relPath} in directory ${codeDir} contains spaces or special characters in its name.`,
                'This might cause problem when uploading to cloud or remote machine.',
                'If you encounter any error, please report an issue: https://github.com/microsoft/nni/issues'
            ].join(' ');
            getLogger('validateCodeDir').warning(message);
        }
    }

    return fileCount;
}
