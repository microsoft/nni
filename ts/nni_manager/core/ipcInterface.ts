// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

export { IpcInterface, createDispatcherInterface } from './tuner_command_channel/shim';
export { createDispatcherPipeInterface, encodeCommand } from './tuner_command_channel/legacy';

export class DummyDispatcherInterface {
    public sendCommand(_commandType: string, _content?: string): void {
        return;
    }

    public onCommand(_listener: (commandType: string, content: string) => void): void {
        return;
    }

    public onError(_listener: (error: Error) => void): void {
        return;
    }
}
