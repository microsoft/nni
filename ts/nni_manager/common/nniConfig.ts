// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

export interface CustomEnvironmentServiceConfig {
    name: string;
    nodeModulePath: string;
    nodeClassName: string;
}

function readConfigFile(fileName: string): string {
    let configPath: string;
    if (process.platform === 'win32') {
        configPath = path.join(process.env.APPDATA as string, 'nni', fileName);
    } else {
        configPath = path.join(os.homedir(), '.config/nni', fileName);
    }
    return fs.readFileSync(configPath).toString();
}

export function getCustomEnvironmentServiceConfig(name: string): CustomEnvironmentServiceConfig | null {
    const config = JSON.parse(readConfigFile('training_services.json'));
    if (config[name] === undefined) {
        return null;
    }
    return {
        name,
        nodeModulePath: config[name].nodeModulePath as string,
        nodeClassName: config[name].nodeClassName as string,
    }
}
