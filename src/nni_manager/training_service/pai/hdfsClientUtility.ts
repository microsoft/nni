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
import { Deferred } from 'ts-deferred';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger } from '../../common/log';
import { unixPathJoin } from '../../common/utils';

/**
 * HDFS client utility, including copy file/directory
 */
export namespace HDFSClientUtility {
    /**
     * Get NNI experiment root directory
     * @param hdfsUserName HDFS user name
     */
    export function hdfsExpRootDir(hdfsUserName: string): string {
        // tslint:disable-next-line:prefer-template
        return '/' + unixPathJoin(hdfsUserName, 'nni', 'experiments', getExperimentId());
    }

    /**
     * Get NNI experiment code directory
     * @param hdfsUserName HDFS user name
     */
    export function getHdfsExpCodeDir(hdfsUserName: string): string {
        return unixPathJoin(hdfsExpRootDir(hdfsUserName), 'codeDir');
    }

    /**
     * Get NNI trial working directory
     * @param hdfsUserName HDFS user name
     * @param trialId NNI trial ID
     */
    export function getHdfsTrialWorkDir(hdfsUserName: string, trialId: string): string {
        const root: string = hdfsExpRootDir(hdfsUserName);

        return unixPathJoin(root, 'trials', trialId);
    }

    /**
     * Copy a local file to hdfs directory
     *
     * @param localFilePath local file path(source)
     * @param hdfsFilePath hdfs file path(target)
     * @param hdfsClient hdfs client
     */
    // tslint:disable: no-unsafe-any non-literal-fs-path no-any
    export async function copyFileToHdfs(localFilePath : string, hdfsFilePath : string, hdfsClient : any) : Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        // tslint:disable-next-line:non-literal-fs-path
        fs.exists(localFilePath, (exists : boolean) => {
            // Detect if local file exist
            if (exists) {
                const localFileStream: fs.ReadStream = fs.createReadStream(localFilePath);
                const hdfsFileStream: any = hdfsClient.createWriteStream(hdfsFilePath);
                localFileStream.pipe(hdfsFileStream);
                hdfsFileStream.on('finish', () => {
                    deferred.resolve();
                });
                hdfsFileStream.on('error', (err : any) => {
                    getLogger()
                      .error(`HDFSCientUtility:copyFileToHdfs, copy file failed, err is ${err.message}`);
                    deferred.reject(err);
                });
            } else {
                getLogger()
                  .error(`HDFSCientUtility:copyFileToHdfs, ${localFilePath} doesn't exist locally`);
                deferred.reject('file not exist!');
            }
        });

        return deferred.promise;
    }

    /**
     * Recursively copy local directory to hdfs directory
     *
     * @param localDirectory local directory
     * @param hdfsDirectory HDFS directory
     * @param hdfsClient   HDFS client
     */
    export async function copyDirectoryToHdfs(localDirectory : string, hdfsDirectory : string, hdfsClient : any) : Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        // TODO: fs.readdirSync doesn't support ~($HOME)
        const fileNameArray: string[] = fs.readdirSync(localDirectory);

        for (const fileName of fileNameArray) {
            const fullFilePath: string = path.join(localDirectory, fileName);
            try {
                // tslint:disable-next-line:non-literal-fs-path
                if (fs.lstatSync(fullFilePath)
                    .isFile()) {
                    await copyFileToHdfs(fullFilePath, path.join(hdfsDirectory, fileName), hdfsClient);
                } else {
                    // If filePath is a directory, recuisively copy it to remote directory
                    await copyDirectoryToHdfs(fullFilePath, path.join(hdfsDirectory, fileName), hdfsClient);
                }
            } catch (error) {
                deferred.reject(error);
            }
        }
        // All files/directories are copied successfully, resolve
        deferred.resolve();

        return deferred.promise;
    }

    /**
     * Read content from HDFS file
     *
     * @param hdfsPath HDFS file path
     * @param hdfsClient HDFS client
     */
    export async function readFileFromHDFS(hdfsPath : string, hdfsClient : any) : Promise<Buffer> {
        const deferred: Deferred<Buffer> = new Deferred<Buffer>();
        let buffer : Buffer = Buffer.alloc(0);

        const exist : boolean = await pathExists(hdfsPath, hdfsClient);
        if (!exist) {
            deferred.reject(`${hdfsPath} doesn't exists`);
        }

        const remoteFileStream: any = hdfsClient.createReadStream(hdfsPath);
        remoteFileStream.on('error', (err : any) => {
            // Reject with the error
            deferred.reject(err);
        });

        remoteFileStream.on('data', (chunk : any) => {
            // Concat the data chunk to buffer
            buffer = Buffer.concat([buffer, chunk]);
        });

        remoteFileStream.on('finish', () => {
            // Upload is done, resolve
            deferred.resolve(buffer);
        });

        return deferred.promise;
    }

    /**
     * Check if an HDFS path already exists
     *
     * @param hdfsPath target path need to check in HDFS
     * @param hdfsClient HDFS client
     */
    export async function pathExists(hdfsPath : string, hdfsClient : any) : Promise<boolean> {
        const deferred : Deferred<boolean> = new Deferred<boolean>();
        hdfsClient.exists(hdfsPath, (exist : boolean) => {
             deferred.resolve(exist);
        });

        let timeoutId : NodeJS.Timer;

        const delayTimeout : Promise<boolean> = new Promise<boolean>((resolve : Function, reject : Function) : void => {
            // Set timeout and reject the promise once reach timeout (5 seconds)
            timeoutId = setTimeout(() => { reject(`Check HDFS path ${hdfsPath} exists timeout`); }, 5000);
        });

        return Promise.race([deferred.promise, delayTimeout])
          .finally(() => { clearTimeout(timeoutId); });
    }

    /**
     * Mkdir in HDFS, use default permission 755
     *
     * @param hdfsPath the path in HDFS. It could be either file or directory
     * @param hdfsClient HDFS client
     */
    export function mkdir(hdfsPath : string, hdfsClient : any) : Promise<boolean> {
        const deferred : Deferred<boolean> = new Deferred<boolean>();

        hdfsClient.mkdir(hdfsPath, (err : any) => {
            if (!err) {
                deferred.resolve(true);
            } else {
                deferred.reject(err.message);
            }
        });

        return deferred.promise;
    }

    /**
     * Read directory contents
     *
     * @param hdfsPath the path in HDFS. It could be either file or directory
     * @param hdfsClient HDFS client
     */
    export async function readdir(hdfsPath : string, hdfsClient : any) : Promise<string[]> {
        const deferred : Deferred<string[]> = new Deferred<string[]>();
        const exist : boolean = await pathExists(hdfsPath, hdfsClient);
        if (!exist) {
            deferred.reject(`${hdfsPath} doesn't exists`);
        }

        hdfsClient.readdir(hdfsPath, (err : any, files : any[]) => {
            if (err) {
                deferred.reject(err);
            }

            deferred.resolve(files);
        });

        return deferred.promise;
    }

    /**
     * Delete HDFS path
     * @param hdfsPath the path in HDFS. It could be either file or directory
     * @param hdfsClient HDFS client
     * @param recursive Mark if need to delete recursively
     */
    export function deletePath(hdfsPath : string, hdfsClient : any, recursive : boolean = true) : Promise<boolean> {
        const deferred : Deferred<boolean> = new Deferred<boolean>();
        hdfsClient.unlink(hdfsPath, recursive, (err : any) => {
            if (!err) {
                deferred.resolve(true);
            } else {
                deferred.reject(err.message);
            }
        });

        return deferred.promise;
    }
    // tslint:enable: no-unsafe-any non-literal-fs-path no-any
}
