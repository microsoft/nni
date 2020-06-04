// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { String } from 'typescript-string-operations';
import { countFilesRecursively, getNewLine, validateFileNameRecursively } from '../../common/utils';
import { GPU_INFO_COLLECTOR_FORMAT_WINDOWS } from './gpuData';

/**
 * Validate codeDir, calculate file count recursively under codeDir, and throw error if any rule is broken
 *
 * @param codeDir codeDir in nni config file
 * @returns file number under codeDir
 */
export async function validateCodeDir(codeDir: string): Promise<number> {
    let fileCount: number | undefined;
    let fileNameValid: boolean = true;
    try {
        fileCount = await countFilesRecursively(codeDir);
    } catch (error) {
        throw new Error(`Call count file error: ${error}`);
    }
    try {
        fileNameValid = await validateFileNameRecursively(codeDir);
    } catch (error) {
        throw new Error(`Validate file name error: ${error}`);
    }

    if (fileCount !== undefined && fileCount > 1000) {
        const errMessage: string = `Too many files(${fileCount} found}) in ${codeDir},`
                                    + ` please check if it's a valid code dir`;
        throw new Error(errMessage);
    }

    if (!fileNameValid) {
        const errMessage: string = `File name in ${codeDir} is not valid, please check file names, only support digit number、alphabet and (.-_) in file name.`;
        throw new Error(errMessage);
    }

    return fileCount;
}

/**
 * crete a new directory
 * @param directory
 */
export async function execMkdir(directory: string, share: boolean = false): Promise<void> {
    if (process.platform === 'win32') {
        await cpp.exec(`powershell.exe New-Item -Path "${directory}" -ItemType "directory" -Force`);
    } else if (share) {
        await cpp.exec(`(umask 0; mkdir -p '${directory}')`);
    } else {
        await cpp.exec(`mkdir -p '${directory}'`);
    }

    return Promise.resolve();
}

/**
 * copy files to the directory
 * @param source
 * @param destination
 */
export async function execCopydir(source: string, destination: string): Promise<void> {
    if (process.platform === 'win32') {
        await cpp.exec(`powershell.exe Copy-Item "${source}\\*" -Destination "${destination}" -Recurse`);
    } else {
        await cpp.exec(`cp -r '${source}/.' '${destination}'`);
    }

    return Promise.resolve();
}

/**
 * crete a new file
 * @param filename
 */
export async function execNewFile(filename: string): Promise<void> {
    if (process.platform === 'win32') {
        await cpp.exec(`powershell.exe New-Item -Path "${filename}" -ItemType "file" -Force`);
    } else {
        await cpp.exec(`touch '${filename}'`);
    }

    return Promise.resolve();
}

/**
 * run script using powershell or bash
 * @param filePath
 */
export function runScript(filePath: string): cp.ChildProcess {
    if (process.platform === 'win32') {
        return cp.exec(`powershell.exe -ExecutionPolicy Bypass -file "${filePath}"`);
    } else {
        return cp.exec(`bash '${filePath}'`);
    }
}

/**
 * output the last line of a file
 * @param filePath
 */
export async function execTail(filePath: string): Promise<cpp.childProcessPromise.Result> {
    let cmdresult: cpp.childProcessPromise.Result;
    if (process.platform === 'win32') {
        cmdresult = await cpp.exec(`powershell.exe Get-Content "${filePath}" -Tail 1`);
    } else {
        cmdresult = await cpp.exec(`tail -n 1 '${filePath}'`);
    }

    return Promise.resolve(cmdresult);
}

/**
 * delete a directory
 * @param directory
 */
export async function execRemove(directory: string): Promise<void> {
    if (process.platform === 'win32') {
        await cpp.exec(`powershell.exe Remove-Item "${directory}" -Recurse -Force`);
    } else {
        await cpp.exec(`rm -rf '${directory}'`);
    }

    return Promise.resolve();
}

/**
 * kill a process
 * @param directory
 */
export async function execKill(pid: string): Promise<void> {
    if (process.platform === 'win32') {
        await cpp.exec(`cmd.exe /c taskkill /PID ${pid} /T /F`);
    } else {
        await cpp.exec(`pkill -P ${pid}`);
    }

    return Promise.resolve();
}

/**
 * get command of setting environment variable
 * @param  variable
 * @returns command string
 */
export function setEnvironmentVariable(variable: { key: string; value: string }): string {
    if (process.platform === 'win32') {
        return `$env:${variable.key}="${variable.value}"`;
    } else {
        return `export ${variable.key}='${variable.value}'`;
    }
}

/**
 * Compress files in directory to tar file
 * @param  sourcePath
 * @param  tarPath
 */
export async function tarAdd(tarPath: string, sourcePath: string): Promise<void> {
    if (process.platform === 'win32') {
        const tarFilePath: string = tarPath.split('\\')
                                    .join('\\\\');
        const sourceFilePath: string = sourcePath.split('\\')
                                   .join('\\\\');
        const script: string[] = [];
        script.push(
            `import os`,
            `import tarfile`,
            String.Format(`tar = tarfile.open("{0}","w:gz")\r\nroot="{1}"\r\nfor file_path,dir,files in os.walk(root):`, tarFilePath, sourceFilePath),
            `    for file in files:`,
            `        full_path = os.path.join(file_path, file)`,
            `        file = os.path.relpath(full_path, root)`,
            `        tar.add(full_path, arcname=file)`,
            `tar.close()`);
        await fs.promises.writeFile(path.join(os.tmpdir(), 'tar.py'), script.join(getNewLine()), { encoding: 'utf8', mode: 0o777 });
        const tarScript: string = path.join(os.tmpdir(), 'tar.py');
        await cpp.exec(`python ${tarScript}`);
    } else {
        await cpp.exec(`tar -czf ${tarPath} -C ${sourcePath} .`);
    }

    return Promise.resolve();
}

/**
 * generate script file name
 * @param fileNamePrefix
 */
export function getScriptName(fileNamePrefix: string): string {
    if (process.platform === 'win32') {
        return String.Format('{0}.ps1', fileNamePrefix);
    } else {
        return String.Format('{0}.sh', fileNamePrefix);
    }
}

export function getGpuMetricsCollectorBashScriptContent(scriptFolder: string): string {
    return `echo $$ > ${scriptFolder}/pid ; METRIC_OUTPUT_DIR=${scriptFolder} python3 -m nni_gpu_tool.gpu_metrics_collector`;
}

export function runGpuMetricsCollector(scriptFolder: string): void {
    if (process.platform === 'win32') {
        const scriptPath = path.join(scriptFolder, 'gpu_metrics_collector.ps1');
        const content = String.Format(GPU_INFO_COLLECTOR_FORMAT_WINDOWS, scriptFolder, path.join(scriptFolder, 'pid'));
        fs.writeFile(scriptPath, content, { encoding: 'utf8' }, () => { runScript(scriptPath); });
    } else {
        cp.exec(getGpuMetricsCollectorBashScriptContent(scriptFolder), { shell: '/bin/bash' });
    }
}
