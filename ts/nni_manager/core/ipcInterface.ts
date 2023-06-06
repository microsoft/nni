// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { IpcInterface, getTunerServer } from './tuner_command_channel';
export { IpcInterface } from './tuner_command_channel';

let tunerDisabled: boolean = false;

export async function createDispatcherInterface(): Promise<IpcInterface> {
    if (!tunerDisabled) {
        return getTunerServer();
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
