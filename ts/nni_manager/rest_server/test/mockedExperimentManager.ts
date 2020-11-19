// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { ExpManager } from '../../common/expmanager';
import { Provider } from 'typescript-ioc';

export const testExpManagerProvider: Provider = {
    get: (): ExpManager => { return new mockedeExpManager(); }
};

export class mockedeExpManager extends ExpManager {
    public getExperimentsInfo(): Promise<JSON> {
        const expInfo = JSON.parse(JSON.stringify({
            "test": {
                "port": 8080,
                "startTime": 1605246730756,
                "endTime": "N/A",
                "status": "RUNNING",
                "platform": "local",
                "experimentName": "testExp",
                "tag": [], "pid": 11111,
                "webuiUrl": [],
                "logDir": null
            }
        }));
        return new Promise<JSON>((resolve, reject) => {
            resolve(expInfo);
        });
    }

    public setExperimentPath(): void {
        return
    }

    public setStatus(): void {
        return
    }

    public stop(): Promise<void> {
        return new Promise<void>(()=>{});
    }
}
