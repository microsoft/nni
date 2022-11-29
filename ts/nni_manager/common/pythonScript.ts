// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { spawn } from 'child_process';
import globals from './globals';
import { Logger, getLogger } from './log';

const logger: Logger = getLogger('pythonScript');

export function runPythonScript(script: string, logTag?: string): Promise<string> {
    return runPython([ '-c', script ], logTag);
}

export function runPythonModule(moduleName: string, args?: string[]): Promise<string> {
    const argsArr = args ?? [];
    return runPython([ '-m', moduleName , ...argsArr ], moduleName);
}

export async function runPython(args: string[], logTag?: string): Promise<string> {
    const proc = spawn(globals.args.pythonInterpreter, args);

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
            logger.warning(`Python command [${logTag}] has stderr:`, stderr);
        } else {
            logger.warning('Python command has stderr.');
            logger.warning('  args:', args);
            logger.warning('  stderr:', stderr);
        }
    }

    return stdout;
}
