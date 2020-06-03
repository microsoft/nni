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

import * as fs from 'fs';
import * as path from 'path';
import { Deferred } from "ts-deferred";
import { StorageService } from "./storageService";

export class MountedStorageService extends StorageService {

    protected internalConfig(_key: string, _value: string): void {
        // nothing to config
    }

    protected async internalRemove(path: string, isDirectory: boolean, isRecursive: boolean): Promise<void> {
        if (isDirectory) {
            if (isRecursive) {
                const children = await fs.promises.readdir(path);
                for (const file of children) {
                    const stat = await fs.promises.lstat(file);
                    this.internalRemove(file, stat.isDirectory(), isRecursive);
                }
            } else {
                await fs.promises.rmdir(path);
            }
        } else {
            await fs.promises.unlink(path);
        }
    }

    protected async internalRename(remotePath: string, newName: string): Promise<void> {
        const dirName = path.dirname(remotePath);
        newName = this.internalJoin(dirName, newName);

        await fs.promises.rename(remotePath, newName);
    }

    protected async internalMkdir(remotePath: string): Promise<void> {
        if (!fs.existsSync(remotePath)) {
            await fs.promises.mkdir(remotePath, { recursive: true });
        }
    }

    protected async internalCopy(sourcePath: string, targetPath: string, isDirectory: boolean, isFromRemote: boolean = false, isToRemote: boolean = true): Promise<string> {
        if (sourcePath === targetPath) {
            return targetPath;
        }

        this.logger.debug(`copying ${sourcePath} to ${targetPath}, dir ${isDirectory}, isFromRemote ${isFromRemote}, isToRemote: ${isToRemote}`);
        if (isDirectory) {
            const basename = isFromRemote ? this.internalBasename(sourcePath) : path.basename(sourcePath);
            if (isToRemote) {
                targetPath = this.internalJoin(targetPath, basename);
                await this.internalMkdir(targetPath);
            } else {
                targetPath = path.join(targetPath, basename);
                await fs.promises.mkdir(targetPath);
            }
            const children = await fs.promises.readdir(sourcePath);
            for (const child of children) {
                const childSourcePath = this.internalJoin(sourcePath, child);
                const stat = await fs.promises.lstat(childSourcePath);
                await this.internalCopy(childSourcePath, targetPath, stat.isDirectory(), isFromRemote, isToRemote);
            }
            return targetPath;
        } else {
            // This behavior may not be consistent for each platform, but it needs to correct to same 
            await this.internalMkdir(targetPath);
            const targetFileName = path.join(targetPath, path.basename(sourcePath));
            await fs.promises.copyFile(sourcePath, targetFileName);
            return targetFileName;
        }
    }

    protected async internalExists(remotePath: string): Promise<boolean> {
        const deferred = new Deferred<boolean>();
        fs.exists(remotePath, (exists) => {
            deferred.resolve(exists);
        });
        return deferred.promise;
    }

    protected async internalRead(remotePath: string, offset?: number, length?: number): Promise<string> {
        const deferred = new Deferred<string>();
        // set a max length to 1MB for performance concern.
        const maxLength = 1024 * 1024;
        if (offset === undefined) {
            offset = -1;
        }
        const current: number = offset < 0 ? 0 : offset;
        if (length === undefined) {
            length = -1;
        }
        const readLength: number = length < 0 ? maxLength : length;
        let result: string = "";

        const stream = fs.createReadStream(remotePath,
            {
                encoding: "utf8",
                start: current,
                end: readLength + current,
            }).on("data", (data) => {
                result += data;
            }).on("end", () => {
                stream.close();
                deferred.resolve(result);
            }).on("error", (err) => {
                deferred.reject(err);
            });

        return deferred.promise;

    }

    protected internalIsRelativePath(remotePath: string): boolean {
        return !path.isAbsolute(remotePath);
    }

    protected internalJoin(...paths: string[]): string {
        return path.join(...paths);
    }

    protected internalDirname(remotePath: string): string {
        return path.dirname(remotePath);
    }

    protected internalBasename(remotePath: string): string {
        return path.basename(remotePath);
    }
}
