// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { spawn } from 'child_process';
import { Logger, getLogger } from './log';
import { getFreePort } from './utils';

const logger: Logger = getLogger('pythonScript');

const python: string = process.platform === 'win32' ? 'python.exe' : 'python3';

interface NNIctlScriptReturnData {
    stdout: string;
    stderr: string;
}

export async function runPythonScript(script: string, logTag?: string): Promise<string> {
    const proc = spawn(python, [ '-c', script ]);

    let stdout: string = '';
    let stderr: string = '';
    proc.stdout.on('data', (data: string) => { stdout += data; });
    proc.stderr.on('data', (data: string) => { stderr += data; });

    const procPromise = new Promise<void>((resolve, reject) => {
        proc.on('error', (err: Error) => { reject(err); });
        proc.on('exit', () => { resolve(); });
    });
    await procPromise;

    if (stderr) {
        if (logTag) {
            logger.warning(`Python script [${logTag}] has stderr:`, stderr);
        } else {
            logger.warning('Python script has stderr.');
            logger.warning('  script:', script);
            logger.warning('  stderr:', stderr);
        }
    }

    return stdout;
}

export async function runNNIctlScript(host: string, ops: string, script: string, logTag?: string): Promise<NNIctlScriptReturnData | null> {
    // const proc = spawn(python, [ '-c', script ]);
    const port = await getFreePort(host, 8080, 10000);
    const proc = spawn(`nnictl`, [ ops, script, '-p', `${port}` ]);

    let stdout: string = '';
    let stderr: string = '';
    proc.stdout.on('data', (data: string) => { stdout += data; });
    proc.stderr.on('data', (data: string) => { stderr += data; });

    const procPromise = new Promise<void>((resolve, reject) => {
        proc.on('error', (err: Error) => { reject(err); });
        proc.on('exit', () => { resolve(); });
    });
    await procPromise;

    if (stderr) {
        if (logTag) {
            logger.warning(`Python script [${logTag}] has stderr:`, stderr);
        } else {
            logger.warning('Python script has stderr.');
            logger.warning('  script:', script);
            logger.warning('  stderr:', stderr);
        }
    }

    return {
        stdout: stdout.trim(),
        stderr: stderr.trim(),
    };
}
