// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { ExperimentManager } from '../../common/experimentManager';
import { Provider } from 'typescript-ioc';

export const testExperimentManagerProvider: Provider = {
    get: (): ExperimentManager => { return new mockedeExperimentManager(); }
};

export class mockedeExperimentManager extends ExperimentManager {
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

    public setExperimentPath(newPath: string): void {
        return
    }

    public setExperimentInfo(experimentId: string, key: string, value: any): void {
        return
    }

    public stop(): Promise<void> {
        return new Promise<void>(()=>{});
    }
}
