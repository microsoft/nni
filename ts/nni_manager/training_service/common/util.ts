// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as cp from 'child_process';
import * as fs from 'fs';
import ignore from 'ignore';
import * as path from 'path';
import * as tar from 'tar';
import { getLogger } from '../../common/log';
import { String } from 'typescript-string-operations';
import { validateFileName } from '../../common/utils';
import { GPU_INFO_COLLECTOR_FORMAT_WINDOWS } from './gpuData';

/**
 * List all files in directory except those ignored by .nniignore.
 * @param source
 * @param destination
 */
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

/**
 * Validate codeDir, calculate file count recursively under codeDir, and throw error if any rule is broken
 *
 * @param codeDir codeDir in nni config file
 * @returns file number under codeDir
 */
export async function validateCodeDir(codeDir: string): Promise<number> {
    let fileCount: number = 0;
    let fileTotalSize: number = 0;
    let fileNameValid: boolean = true;
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
        fileNameValid = true;
        relPath.split(path.sep).forEach(fpart => {
            if (fpart !== '' && !validateFileName(fpart))
                fileNameValid = false;
        });
        if (!fileNameValid) {
            throw new Error(`Validate file name error: '${d}' is an invalid file name.`);
        }
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
    if (!fs.existsSync(destination))
        await fs.promises.mkdir(destination);
    for (const relPath of listDirWithIgnoredFiles(source, '', [])) {
        const sourcePath = path.join(source, relPath);
        const destPath = path.join(destination, relPath);
        if (fs.statSync(sourcePath).isDirectory()) {
            if (!fs.existsSync(destPath)) {
                await fs.promises.mkdir(destPath);
            }
        } else {
            getLogger().debug(`Copying file from ${sourcePath} to ${destPath}`);
            await fs.promises.copyFile(sourcePath, destPath);
        }
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
    return `echo $$ > ${scriptFolder}/pid ; METRIC_OUTPUT_DIR=${scriptFolder} python3 -m nni.tools.gpu_tool.gpu_metrics_collector`;
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
