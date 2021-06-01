// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { spawn } from 'child_process';
import { Logger, getLogger } from './log';

const python = process.platform === 'win32' ? 'python.exe' : 'python3';

export async function runPythonScript(script: string, logger?: Logger): Promise<string> {
    const proc = spawn(python, [ '-c', script ]);

    const procPromise = new Promise<void>((resolve, reject) => {
        proc.on('error', (err: Error) => { reject(err); });
        proc.on('exit', () => { resolve(); });
    });
    await procPromise;

    const stdout = proc.stdout.read().toString();
    const stderr = proc.stderr.read().toString();

    if (stderr) {
        if (logger === undefined) {
            logger = getLogger('pythonScript');
        }
        logger.warning('python script has stderr.');
        logger.warning('script:', script);
        logger.warning('stderr:', stderr);
    }

    return stdout;
}
