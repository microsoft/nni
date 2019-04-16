import { getLogger } from "common/log";

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

import { countFilesRecursively } from '../../common/utils'
import * as cpp from 'child-process-promise';
import * as cp from 'child_process';
import { GPU_INFO_COLLECTOR_FORMAT_LINUX, GPU_INFO_COLLECTOR_FORMAT_WINDOWS } from './gpuData'
import * as path from 'path';
import { String } from 'typescript-string-operations';
import { file } from "../../node_modules/@types/tmp";

/**
 * Validate codeDir, calculate file count recursively under codeDir, and throw error if any rule is broken
 * 
 * @param codeDir codeDir in nni config file
 * @returns file number under codeDir
 */
export async function validateCodeDir(codeDir: string) : Promise<number> {
    let fileCount: number | undefined;

    try {
        fileCount = await countFilesRecursively(codeDir);
    } catch(error) {
        throw new Error(`Call count file error: ${error}`);
    }

    if(fileCount && fileCount > 1000) {
        const errMessage: string = `Too many files(${fileCount} found}) in ${codeDir},` 
                                    + ` please check if it's a valid code dir`;
        throw new Error(errMessage);        
    }

    return fileCount;
}

/**
 * crete a new directory
 * @param directory 
 */
export async function execMkdir(directory: string): Promise<void> {
    if (process.platform === 'win32') {
        await cpp.exec(`powershell.exe New-Item -Path ${directory} -ItemType "directory" -Force`);
    } else {
        await cpp.exec(`mkdir -p ${directory}`);
    }
    return Promise.resolve();
}

/**
 * crete a new file
 * @param filename 
 */
export async function execNewFile(filename: string): Promise<void> {
    if (process.platform === 'win32') {
        await cpp.exec(`powershell.exe New-Item -Path ${filename} -ItemType "file" -Force`);
    } else {
        await cpp.exec(`touch ${filename}`);
    }
    return Promise.resolve();
}

/**
 * run script
 * @param filePath
 */
export function execScript(filePath: string): cp.ChildProcess {
    if (process.platform === 'win32') {
        return cp.exec(`powershell.exe -file ${filePath}`);
    } else {
        return cp.exec(`bash ${filePath}`);
    }
}



/**
 * output the last line of a file
 * @param filePath 
 */
export async function execTail(filePath: string): Promise<cpp.childProcessPromise.Result> {
    let cmdresult: cpp.childProcessPromise.Result;
    if (process.platform === 'win32') {
        cmdresult = await cpp.exec(`powershell.exe Get-Content ${filePath} -Tail 1`);
    } else {
        cmdresult = await cpp.exec(`tail -n 1 ${filePath}`);
    }
    return Promise.resolve(cmdresult);
}

/**
 * delete a directory
 * @param directory 
 */
export async function execRemove(directory: string): Promise<void>{
    if (process.platform === 'win32') {
        await cpp.exec(`powershell.exe Remove-Item ${directory}`);
    } else {
        await cpp.exec(`rm -rf ${directory}`);
    }
    return Promise.resolve();
}

/**
 * kill a process
 * @param directory 
 */
export async function execKill(pid: string): Promise<void>{
    if (process.platform === 'win32') {
        await cpp.exec(`cmd /c taskkill /PID ${pid} /T /F`);
    } else {
        await cpp.exec(`pkill -P ${pid}`);
    }
    return Promise.resolve();
}

/**
 * set environment variable
 * @param  variable
 * @returns command string  
 */
export function setEnvironmentVariable(variable: { key: string; value: string }): string{
    if (process.platform === 'win32') {
        return `$env:${variable.key}="${variable.value}"`;
    }
    else{
        return `export ${variable.key}=${variable.value}`;
    }
}


/**
 * generate script file name
 * @param fileNamePrefix 
 */
export function getScriptName(fileNamePrefix: string): string {
    if (process.platform === 'win32') {
        return fileNamePrefix + '.ps1';
    } else {
        return fileNamePrefix + '.sh';
    }
}

/**
 * generate script file
 * @param gpuMetricCollectorScriptFolder 
 */
export function getgpuMetricsCollectorScriptContent(gpuMetricCollectorScriptFolder: string): string {
    if(process.platform === 'win32') {
        return String.Format(
            GPU_INFO_COLLECTOR_FORMAT_WINDOWS,
            gpuMetricCollectorScriptFolder,
            path.join(gpuMetricCollectorScriptFolder, 'pid'),
        );
    } else {
        return String.Format(
            GPU_INFO_COLLECTOR_FORMAT_LINUX,
            gpuMetricCollectorScriptFolder,
            path.join(gpuMetricCollectorScriptFolder, 'pid'),
        );
    }
}
