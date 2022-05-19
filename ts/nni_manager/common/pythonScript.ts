// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { spawn } from 'child_process';
import globals from './globals';
import { Logger, getLogger } from './log';

const logger: Logger = getLogger('pythonScript');

export async function runPythonScript(script: string, logTag?: string): Promise<string> {
    const proc = spawn(globals.args.pythonInterpreter, [ '-c', script ]);

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
