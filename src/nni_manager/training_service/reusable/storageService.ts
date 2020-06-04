/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import { uniqueString } from '../../common/utils';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Logger, getLogger } from '../../common/log';
import { tarAdd } from '../common/util';

export abstract class StorageService {
    protected localRoot: string = "";
    protected remoteRoot: string = "";
    protected logger: Logger;

    protected abstract internalConfig(key: string, value: string): void;
    protected abstract async internalRemove(remotePath: string, isDirectory: boolean, isRecursive: boolean): Promise<void>;
    protected abstract async internalRename(remotePath: string, newName: string): Promise<void>;
    protected abstract async internalMkdir(remotePath: string): Promise<void>;
    protected abstract async internalCopy(sourcePath: string, targetPath: string, isDirectory: boolean, isFromRemote: boolean, isToRemote: boolean): Promise<string>;
    protected abstract async internalExists(remotePath: string): Promise<boolean>;
    protected abstract async internalRead(remotePath: string, offset: number, length: number): Promise<string>;
    protected abstract async internalList(remotePath: string): Promise<string[]>;
    protected abstract internalIsRelativePath(path: string): boolean;
    protected abstract internalJoin(...paths: string[]): string;
    protected abstract internalDirname(...paths: string[]): string;
    protected abstract internalBasename(...paths: string[]): string;

    constructor() {
        this.logger = getLogger();
    }

    public initialize(localRoot: string, remoteRoot: string): void {
        this.logger.debug(`Initializing storage to local: ${localRoot} remote: ${remoteRoot}`);
        this.localRoot = localRoot;
        this.remoteRoot = remoteRoot;
    }

    public async rename(remotePath: string, newName: string): Promise<void> {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`rename remotePath: ${remotePath} to: ${newName}`);
        await this.internalRename(remotePath, newName);
    }

    public async createDirectory(remotePath: string): Promise<void> {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`create remotePath: ${remotePath}`);
        await this.internalMkdir(remotePath);
    }

    public async copyDirectory(localPath: string, remotePath: string, asGzip: boolean = false): Promise<string> {
        localPath = this.expandPath(false, localPath);
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`copy localPath: ${localPath} to remotePath: ${remotePath}, asGzip ${asGzip}`);
        if (!await this.exists(remotePath)) {
            await this.internalMkdir(remotePath);
        }

        if (asGzip) {
            const localPathBaseName = path.basename(localPath);
            const tempTarFileName = `nni_tmp_${localPathBaseName}_${uniqueString(5)}.tar.gz`;
            const tarFileName = `${localPathBaseName}.tar.gz`;
            const localTarPath: string = path.join(os.tmpdir(), tempTarFileName);
            await tarAdd(localTarPath, localPath);
            await this.internalCopy(localTarPath, remotePath, false, false, true);
            const remoteFileName = this.internalJoin(remotePath, tempTarFileName);
            await this.internalRename(remoteFileName, tarFileName);
            await fs.promises.unlink(localTarPath);

            remotePath = this.internalJoin(remotePath, tarFileName);
        } else {
            await this.internalCopy(localPath, remotePath, true, false, true);
            remotePath = this.internalJoin(remotePath, path.basename(localPath));
        }

        return remotePath;
    }

    public async copyDirectoryBack(remotePath: string, localPath: string): Promise<string> {
        localPath = this.expandPath(false, localPath);
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`copy remotePath: ${remotePath} to localPath: ${localPath}`);
        return await this.internalCopy(localPath, remotePath, true, true, false);
    }

    public async removeDirectory(remotePath: string, isRecursive: boolean): Promise<void> {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`remove remotePath: ${remotePath}`);
        await this.internalRemove(remotePath, true, isRecursive);
    }

    public async readFileContent(remotePath: string, offset: number = -1, length: number = -1): Promise<string> {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`read remote file: ${remotePath}, offset: ${offset}, length: ${length}`);
        return this.internalRead(remotePath, offset, length);
    }

    public async listDirectory(remotePath: string): Promise<string[]> {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`list remotePath: ${remotePath}`);
        return await this.internalList(remotePath);
    }

    public async exists(remotePath: string): Promise<boolean> {
        remotePath = this.expandPath(true, remotePath);
        const exists = await this.internalExists(remotePath);
        this.logger.debug(`exists remotePath: ${remotePath} is ${exists}`);
        return exists
    }

    public async save(content: string, remotePath: string): Promise<void> {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`saving content to remotePath: ${remotePath}, length: ${content.length}`);
        const fileName = this.internalBasename(remotePath);
        const tempFileName = `temp_${uniqueString(4)}_${fileName}`;

        const localTempFileName = path.join(os.tmpdir(), tempFileName);

        const remoteDir = this.internalDirname(remotePath);
        const remoteTempFile = this.internalJoin(remoteDir, tempFileName);

        if (await this.internalExists(remotePath) === true) {
            await this.internalRemove(remotePath, false, false);
        }
        await fs.promises.writeFile(localTempFileName, content);
        await this.internalCopy(localTempFileName, remoteDir, false, false, true);
        await this.rename(remoteTempFile, fileName);
        await fs.promises.unlink(localTempFileName);
    }

    public async copyFile(localPath: string, remotePath: string): Promise<void> {
        localPath = this.expandPath(false, localPath);
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`copying file localPath: ${localPath} to remotePath: ${remotePath}`);
        await this.internalCopy(localPath, remotePath, false, false, true);
    }

    public async copyFileBack(remotePath: string, localPath: string): Promise<void> {
        localPath = this.expandPath(false, localPath);
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`copy file remotePath: ${remotePath} to localPath: ${localPath}`);
        await this.internalCopy(localPath, remotePath, false, true, false);
    }

    public async removeFile(remotePath: string): Promise<void> {
        remotePath = this.expandPath(true, remotePath);
        this.logger.debug(`remove file remotePath: ${remotePath}`);
        await this.internalRemove(remotePath, false, false);
    }

    public joinPath(...paths: string[]): string {
        let fullPath = this.internalJoin(...paths);
        if (this.internalIsRelativePath(fullPath) === true && this.remoteRoot !== "") {
            fullPath = this.internalJoin(this.remoteRoot, fullPath);
        }
        return fullPath;
    }

    private expandPath(isRemote: boolean, ...paths: string[]): string {
        let normalizedPath: string;

        if (isRemote) {
            normalizedPath = this.joinPath(...paths);
        } else {
            normalizedPath = path.join(...paths);
            if (!path.isAbsolute(normalizedPath) && this.localRoot !== "") {
                normalizedPath = path.join(this.localRoot, normalizedPath);
            }
        }

        return normalizedPath;
    }
}
