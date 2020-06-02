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
import { StorageService } from "./storage";

export class MountedStorageService extends StorageService {

    protected config(_key: string, _value: string): void {
        // nothing to config
    }

    protected async remove(path: string, isDirectory: boolean, isRecursive: boolean): Promise<void> {
        if (isDirectory) {
            if (isRecursive) {
                const children = await fs.promises.readdir(path);
                for (const file of children) {
                    const stat = await fs.promises.lstat(file);
                    this.remove(file, stat.isDirectory(), isRecursive);
                }
            } else {
                await fs.promises.rmdir(path);
            }
        } else {
            await fs.promises.unlink(path);
        }
    }

    protected async rename(remotePath: string, newName: string): Promise<void> {
        const dirName = path.dirname(remotePath);
        newName = this.joinPath(dirName, newName);

        await fs.promises.rename(remotePath, newName);
    }

    protected async mkdir(remotePath: string): Promise<void> {
        if (!fs.existsSync(remotePath)) {
            await fs.promises.mkdir(remotePath, { recursive: true });
        }
    }

    protected async copy(localPath: string, remotePath: string, isDirectory: boolean, isToRemote: boolean): Promise<string> {
        if (localPath === remotePath) {
            return remotePath;
        }
        const sourcePath = isToRemote ? localPath : remotePath;
        let targetPath = isToRemote ? remotePath : localPath;

        this.logger.debug(`copying ${sourcePath} to ${targetPath}, dir ${isDirectory}, isRemote: ${isToRemote}`);
        if (isDirectory) {
            if (isToRemote) {
                targetPath = this.joinPath(targetPath, this.basename(localPath));
            } else {
                targetPath = path.join(targetPath, this.basename(remotePath))
            }
            await this.mkdir(targetPath);
            const children = await fs.promises.readdir(sourcePath);
            for (const child of children) {
                const childSourcePath = this.joinPath(sourcePath, child);
                const stat = await fs.promises.lstat(childSourcePath);
                // true: the source and target is aligned already, so always set isToRemote to true.
                this.copy(childSourcePath, targetPath, stat.isDirectory(), true);
            }
            return targetPath;
        } else {
            // This behavior may not be consistent for each platform, but it needs to correct to same 
            await this.mkdir(targetPath);
            const targetFileName = path.join(targetPath, path.basename(sourcePath));
            await fs.promises.copyFile(sourcePath, targetFileName);
            return targetFileName;
        }
    }

    protected async exists(remotePath: string): Promise<boolean> {
        const deferred = new Deferred<boolean>();
        fs.exists(remotePath, (exists) => {
            deferred.resolve(exists);
        });
        return deferred.promise;
    }

    protected async read(remotePath: string, offset?: number, length?: number): Promise<string> {
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

    protected isRelativePath(remotePath: string): boolean {
        return !path.isAbsolute(remotePath);
    }

    protected joinPath(...paths: string[]): string {
        return path.join(...paths);
    }

    protected dirname(remotePath: string): string {
        return path.dirname(remotePath);
    }

    protected basename(remotePath: string): string {
        return path.basename(remotePath);
    }
}
