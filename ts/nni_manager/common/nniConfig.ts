// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'fs';
import path from 'path';
import { promisify } from 'util';
import { runPythonScript } from './pythonScript';

export interface CustomEnvironmentServiceConfig {
    name: string;
    nodeModulePath: string;
    nodeClassName: string;
}

const readFile = promisify(fs.readFile);

async function readConfigFile(fileName: string): Promise<string> {
    const script = 'import nni.runtime.config ; print(nni.runtime.config.get_config_directory())';
    const configDir = (await runPythonScript(script)).trim();
    const stream = await readFile(path.join(configDir, fileName));
    return stream.toString();
}

export async function getCustomEnvironmentServiceConfig(name: string): Promise<CustomEnvironmentServiceConfig | null> {
    const configJson = await readConfigFile('training_services.json');
    const config = JSON.parse(configJson);
    if (config[name] === undefined) {
        return null;
    }
    return {
        name,
        nodeModulePath: config[name].nodeModulePath as string,
        nodeClassName: config[name].nodeClassName as string,
    }
}
