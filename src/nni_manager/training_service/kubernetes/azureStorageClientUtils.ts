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

import * as fs from 'fs'
import * as path from 'path';
import { Deferred } from 'ts-deferred';
import { getLogger } from '../../common/log';
import { mkDirP } from '../../common/utils';

export namespace AzureStorageClientUtility {
    
    /**
     * create azure share
     * @param fileServerClient 
     * @param azureShare 
     */
    export async function createShare(fileServerClient: any, azureShare: any): Promise<void>{
        const deferred: Deferred<void> = new Deferred<void>();
        fileServerClient.createShareIfNotExists(azureShare, function(error: any, result: any, response: any) {
            if(error){
                getLogger().error(`Create share failed:, ${error}`);
                deferred.reject(error)
            }else{
                deferred.resolve()
            }
        })
        return deferred.promise;
    }
    
    /**
     * Create a new directory (NOT recursively) in azure file storage.
     * @param fileServerClient 
     * @param azureFoler 
     * @param azureShare 
     */
    export async function createDirectory(fileServerClient: any, azureFoler: any, azureShare: any): Promise<void>{
        const deferred: Deferred<void> = new Deferred<void>();
        fileServerClient.createDirectoryIfNotExists(azureShare, azureFoler, function(error: any, result: any, response: any) {
            if(error){
                getLogger().error(`Create directory failed:, ${error}`);
                deferred.reject(error);
            }else{
                deferred.resolve();
            }
        })
        return deferred.promise;
    }

    /**
     * Create a new directory recursively in azure file storage
     * @param fileServerClient
     * @param azureDirectory 
     */
    export async function createDirectoryRecursive(fileServerClient: any, azureDirectory: any, azureShare: any): Promise<void>{
        const deferred: Deferred<void> = new Deferred<void>();
        let directories = azureDirectory.split("/");
        let rootDirectory = ""
        for(let directory of directories){
            rootDirectory += directory;
            await createDirectory(fileServerClient, rootDirectory, azureShare);
            rootDirectory += '/';
        }
        deferred.resolve();
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
    async function uploadFileToAzure(fileServerClient: any, azureDirectory: any, azureFileName: any, azureShare: any, localFilePath: any): Promise<void>{
        const deferred: Deferred<void> = new Deferred<void>();
        await fileServerClient.createFileFromLocalFile(azureShare, azureDirectory, azureFileName, localFilePath, function(error: any, result: any, response: any) {
            if(error){
                getLogger().error(`Upload file failed:, ${error}`);
                deferred.reject(error);
            }else{          
                deferred.resolve();
            }
        })
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
    async function downloadFile(fileServerClient: any, azureDirectory: any, azureFileName: any, azureShare: any, localFilePath: any): Promise<void>{
        const deferred: Deferred<void> = new Deferred<void>();
        await fileServerClient.getFileToStream(azureShare, azureDirectory, azureFileName, fs.createWriteStream(localFilePath), function(error: any, result: any, response: any) {
            if(error){
                getLogger().error(`Download file failed:, ${error}`);
                deferred.reject(error);
            }else{
                deferred.resolve();    
            }
        })
        return deferred.promise;
    }

    /**
     * Upload a directory to azure file storage
     * @param fileServerClient : the client of file server
     * @param azureDirectory : the directory in azure file storage
     * @param azureShare : the azure share used
     * @param localDirectory : local directory to be uploaded
     */
    export async function uploadDirectory(fileServerClient: any, azureDirectory: any, azureShare: any, localDirectory: any): Promise<void>{
        const deferred: Deferred<void> = new Deferred<void>();
        const fileNameArray: string[] = fs.readdirSync(localDirectory);
        await createDirectoryRecursive(fileServerClient, azureDirectory, azureShare);
        for(let fileName of fileNameArray){
            const fullFilePath: string = path.join(localDirectory, fileName);
            try {
                if (fs.lstatSync(fullFilePath).isFile()) {
                    await uploadFileToAzure(fileServerClient, azureDirectory, fileName, azureShare, fullFilePath);
                } else {
                    // If filePath is a directory, recuisively copy it to azure
                    await uploadDirectory(fileServerClient, azureDirectory + '/' + fileName, azureShare, fullFilePath);
                }
            } catch(error) {
                deferred.reject(error);
                return deferred.promise;
            }
        }
        // All files/directories are copied successfully, resolve
        deferred.resolve();
        return deferred.promise;
    }
    
    /**
     * downlod a directory from azure
     * @param fileServerClient 
     * @param azureDirectory 
     * @param azureShare 
     * @param localDirectory 
     */
    export async function downloadDirectory(fileServerClient: any, azureDirectory:any, azureShare: any, localDirectory: any): Promise<void>{
        const deferred: Deferred<void> = new Deferred<void>();
        mkDirP(localDirectory);
        fileServerClient.listFilesAndDirectoriesSegmented(azureShare, azureDirectory, 'null', function(error: any, result: any, response: any) {
            if(('entries' in result) === false){
                getLogger().error(`list files failed, can't get entries in result`);
                throw new Error(`list files failed, can't get entries in result`);
            }

            if(('files' in result['entries']) === false){
                getLogger().error(`list files failed, can't get files in result['entries']`);
                throw new Error(`list files failed, can't get files in result['entries']`);
            }

            if(('directories' in result['directories']) === false){
                getLogger().error(`list files failed, can't get directories in result['entries']`);
                throw new Error(`list files failed, can't get directories in result['entries']`);
            }

            for(var fileName of result['entries']['files']){
                const fullFilePath: string = path.join(localDirectory, fileName.name);
                downloadFile(fileServerClient, azureDirectory, fileName.name, azureShare, fullFilePath)
            }
            
            for(var directoryName of result['entries']['directories']){
                const fullDirectoryPath: string = path.join(localDirectory, directoryName.name)
                const fullAzureDirectory: string = path.join(azureDirectory, directoryName.name)
                downloadDirectory(fileServerClient, fullAzureDirectory, azureShare, fullDirectoryPath)
            }
            deferred.resolve();
        })
        return deferred.promise;
    }
}
