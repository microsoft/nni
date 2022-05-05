// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { IpcInterface } from './tuner_command_channel/common';
export { IpcInterface } from './tuner_command_channel/common';
import * as shim from './tuner_command_channel/shim';

let tunerDisabled: boolean = false;

export async function createDispatcherInterface(): Promise<IpcInterface> {
    if (!tunerDisabled) {
        return await shim.createDispatcherInterface();
    } else {
        return new DummyIpcInterface();
    }
}

export function encodeCommand(commandType: string, content: string): Buffer {
    const contentBuffer: Buffer = Buffer.from(content);
    const contentLengthBuffer: Buffer = Buffer.from(contentBuffer.length.toString().padStart(14, '0'));
    return Buffer.concat([Buffer.from(commandType), contentLengthBuffer, contentBuffer]);
}

class DummyIpcInterface implements IpcInterface {
    public async init(): Promise<void> { /* empty */ }
    public sendCommand(_commandType: string, _content?: string): void { /* empty */ }
    public onCommand(_listener: (commandType: string, content: string) => void): void { /* empty */ }
    public onError(_listener: (error: Error) => void): void { /* empty */ }
}

export namespace UnitTestHelpers {
    export function disableTuner(): void {
        tunerDisabled = true;
    }
}
