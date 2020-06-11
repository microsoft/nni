// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as azureStorage from 'azure-storage';
import * as fs from 'fs';
import * as path from 'path';
import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import { getLogger } from '../../common/log';
import { mkDirP } from '../../common/utils';

export namespace AzureStorageClientUtility {

    /**
     * create azure share
     * @param fileServerClient
     * @param azureShare
     */
    export async function createShare(fileServerClient: any, azureShare: any): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        fileServerClient.createShareIfNotExists(azureShare, (error: any, _result: any, _response: any) => {
            if (error) {
                getLogger()
                  .error(`Create share failed:, ${error}`);
                deferred.resolve(false);
            } else {
                deferred.resolve(true);
            }
        });

        return deferred.promise;
    }

    /**
     * Create a new directory (NOT recursively) in azure file storage.
     * @param fileServerClient
     * @param azureFoler
     * @param azureShare
     */
    export async function createDirectory(fileServerClient: azureStorage.FileService, azureFoler: any, azureShare: any): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        fileServerClient.createDirectoryIfNotExists(azureShare, azureFoler, (error: any, _result: any, _response: any) => {
            if (error) {
                getLogger()
                  .error(`Create directory failed:, ${error}`);
                deferred.resolve(false);
            } else {
                deferred.resolve(true);
            }
        });
        return deferred.promise;
    }

    /**
     * Create a new directory recursively in azure file storage
     * @param fileServerClient
     * @param azureDirectory
     */
    export async function createDirectoryRecursive(fileServerClient: azureStorage.FileService, azureDirectory: string,
                                                   azureShare: any): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        const directories: string[] = azureDirectory.split('/');
        let rootDirectory: string = '';
        for (const directory of directories) {
            rootDirectory += directory;
            const result: boolean = await createDirectory(fileServerClient, rootDirectory, azureShare);
            if (!result) {
                deferred.resolve(false);
                return deferred.promise;
            }
            rootDirectory += '/';
        }
        deferred.resolve(true);

        return deferred.promise;
    }

    /**
     * upload a file to azure storage
     * @param fileServerClient
     * @param azureDirectory
     * @param azureFileName
     * @param azureShare
     * @param localFilePath
     */
    async function uploadFileToAzure(fileServerClient: any, azureDirectory: string, azureFileName: any, azureShare: any,
                                     localFilePath: string): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        await fileServerClient.createFileFromLocalFile(azureShare, azureDirectory, azureFileName, localFilePath,
                                                       (error: any, _result: any, _response: any) => {
            if (error) {
                getLogger()
                  .error(`Upload file failed:, ${error}`);
                deferred.resolve(false);
            } else {
                deferred.resolve(true);
            }
        });

        return deferred.promise;
    }

    /**
     * download a file from azure storage
     * @param fileServerClient
     * @param azureDirectory
     * @param azureFileName
     * @param azureShare
     * @param localFilePath
     */
    async function downloadFile(fileServerClient: any, azureDirectory: string, azureFileName: any, azureShare: any,
                                localFilePath: string): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        await fileServerClient.getFileToStream(azureShare, azureDirectory, azureFileName, fs.createWriteStream(localFilePath),
                                               (error: any, _result: any, _response: any) => {
            if (error) {
                getLogger()
                  .error(`Download file failed:, ${error}`);
                deferred.resolve(false);
            } else {
                deferred.resolve(true);
            }
        });

        return deferred.promise;
    }

    /**
     * Upload a directory to azure file storage
     * @param fileServerClient : the client of file server
     * @param azureDirectory : the directory in azure file storage
     * @param azureShare : the azure share used
     * @param localDirectory : local directory to be uploaded
     */
    export async function uploadDirectory(fileServerClient: azureStorage.FileService, azureDirectory: string, azureShare: any,
                                          localDirectory: string): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        const fileNameArray: string[] = fs.readdirSync(localDirectory);
        const result: boolean = await createDirectoryRecursive(fileServerClient, azureDirectory, azureShare);
        if (!result) {
            deferred.resolve(false);
            return deferred.promise;
        }
        for (const fileName of fileNameArray) {
            const fullFilePath: string = path.join(localDirectory, fileName);
            try {
                let resultUploadFile: boolean = true;
                let resultUploadDir: boolean = true;
                if (fs.lstatSync(fullFilePath)
                      .isFile()) {
                    resultUploadFile = await uploadFileToAzure(fileServerClient, azureDirectory, fileName, azureShare, fullFilePath);
                } else {
                    // If filePath is a directory, recuisively copy it to azure
                    resultUploadDir = await uploadDirectory(fileServerClient, String.Format('{0}/{1}', azureDirectory, fileName), azureShare, fullFilePath);
                }
                if (!(resultUploadFile && resultUploadDir)) {
                    deferred.resolve(false);
                    return deferred.promise;
                }
            } catch (error) {
                deferred.resolve(false);

                return deferred.promise;
            }
        }
        // All files/directories are copied successfully, resolve
        deferred.resolve(true);

        return deferred.promise;
    }

    /**
     * downlod a directory from azure
     * @param fileServerClient
     * @param azureDirectory
     * @param azureShare
     * @param localDirectory
     */
    export async function downloadDirectory(fileServerClient: any, azureDirectory: string, azureShare: any, localDirectory: string):
     Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        await mkDirP(localDirectory);
        fileServerClient.listFilesAndDirectoriesSegmented(azureShare, azureDirectory, 'null',
                                                          async (_error: any, result: any, _response: any) => {
            if (('entries' in result) === false) {
                getLogger()
                  .error(`list files failed, can't get entries in result`);
                throw new Error(`list files failed, can't get entries in result`);
            }

            if (('files' in result.entries) === false) {
                getLogger()
                  .error(`list files failed, can't get files in result['entries']`);
                throw new Error(`list files failed, can't get files in result['entries']`);
            }

            if (('directories' in result.directories) === false) {
                getLogger()
                  .error(`list files failed, can't get directories in result['entries']`);
                throw new Error(`list files failed, can't get directories in result['entries']`);
            }

            for (const fileName of result.entries.files) {
                const fullFilePath: string = path.join(localDirectory, fileName.name);
                await downloadFile(fileServerClient, azureDirectory, fileName.name, azureShare, fullFilePath);
            }

            for (const directoryName of result.entries.directories) {
                const fullDirectoryPath: string = path.join(localDirectory, directoryName.name);
                const fullAzureDirectory: string = path.join(azureDirectory, directoryName.name);
                await downloadDirectory(fileServerClient, fullAzureDirectory, azureShare, fullDirectoryPath);
            }
            deferred.resolve();
        });

        return deferred.promise;
    }
}
