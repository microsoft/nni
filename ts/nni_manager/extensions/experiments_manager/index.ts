// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { ExperimentsManager } from './manager';
export { ExperimentsManager } from './manager';

let singleton: ExperimentsManager | null = null;

export function initExperimentsManager(): void {
    getExperimentsManager();
}

export function getExperimentsManager(): ExperimentsManager {
    if (singleton === null) {
        singleton = new ExperimentsManager();
    }
    return singleton;
}

export namespace UnitTestHelpers {
    export function setExperimentsManager(experimentsManager: any): void {
        singleton = experimentsManager;
    }
}
