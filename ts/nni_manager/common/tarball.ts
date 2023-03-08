// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import type { Stats } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';

import ignore, { Ignore } from 'ignore';
import tar from 'tar';

import { globals } from 'common/globals';
import { Logger, getLogger } from 'common/log';

const logger: Logger = getLogger('common.tarball');

export async function createTarball(tarName: string, sourcePath: string): Promise<string> {
    const fileList = [];

    let ignorePatterns;
    try {
        ignorePatterns = await fs.readFile(path.join(sourcePath, '.nniignore'), { encoding: 'utf8' });
    } catch { }
    const ig = ignorePatterns ? ignore().add(ignorePatterns) : undefined;

    let countNum = 0;
    let countSize = 0;

    for await (const [file, stats] of walk(sourcePath, '', ig)) {
        if (stats.isSymbolicLink()) {
            logger.warning(`${sourcePath} contains a symlink ${file}. It will be uploaded as is and might not work`);
        }

        fileList.push(file);
        countNum += 1;
        countSize += stats.size;

        if (countNum > 2000) {
            logger.error(`Failed to pack ${sourcePath}: too many files`);
            throw new Error(`${sourcePath} contains too many files (more than 2000)`);
        }
        if (countSize > 1024 * 1024 * 1024) {
            logger.error(`Failed to pack ${sourcePath}: too large`);
            throw new Error(`${sourcePath} is too large (more than 1GB)`);
        }
    }

    // TODO: move it to globals.paths
    const tarDir = path.join(globals.paths.experimentRoot, 'tarball');
    await fs.mkdir(tarDir, { recursive: true });
    const tarPath = path.join(tarDir, `${tarName}.tgz`);

    const opts = {
        gzip: true,
        file: tarPath,
        cwd: sourcePath,
        portable: true,
    } as const;
    await tar.create(opts, fileList);

    return tarPath;
}

async function* walk(root: string, relDir: string, ig?: Ignore): AsyncGenerator<[string, Stats]> {
    const dir = path.join(root, relDir);
    const entries = await fs.readdir(dir);

    for (const entry of entries) {
        const stats = await fs.lstat(path.join(dir, entry));
        const relEntry = path.join(relDir, entry);

        if (ig && ig.ignores(relEntry + (stats.isDirectory() ? path.sep : ''))) {
            continue;
        }

        if (stats.isDirectory()) {
            yield* walk(root, relEntry, ig);
        }  else {
            yield [relEntry, stats];
        }
    }
}
