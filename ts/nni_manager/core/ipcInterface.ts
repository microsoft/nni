// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { IpcInterface } from './tuner_command_channel/common';
export { IpcInterface } from './tuner_command_channel/common';
export { createDispatcherPipeInterface, encodeCommand } from './tuner_command_channel/legacy';
import * as shim from './tuner_command_channel/shim';

let tunerDisabled: boolean = false;

class DummyIpcInterface implements IpcInterface {
    public sendCommand(_commandType: string, _content?: string): void { /* empty */ }
    public onCommand(_listener: (commandType: string, content: string) => void): void { /* empty */ }
    public onError(_listener: (error: Error) => void): void { /* empty */ }
}

export async function createDispatcherInterface(): Promise<IpcInterface> {
    if (!tunerDisabled) {
        return await shim.createDispatcherInterface();
    } else {
        return new DummyIpcInterface();
    }
}

export namespace UnitTestHelpers {
    export function disableTuner(): void {
        tunerDisabled = true;
    }
}
