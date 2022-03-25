// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import * as CommandType from './commands';

export { IpcInterface, createDispatcherInterface } from './tuner_command_channel/shim';

/**
 * Encode a command
 * @param commandType a command type defined in 'core/commands'
 * @param content payload of the command
 * @returns binary command data
 */
export function encodeCommand(commandType: string, content: string): Buffer {
    const contentBuffer: Buffer = Buffer.from(content);
    const contentLengthBuffer: Buffer = Buffer.from(contentBuffer.length.toString().padStart(14, '0'));
    return Buffer.concat([Buffer.from(commandType), contentLengthBuffer, contentBuffer]);
}

export class DummyDispatcherInterface {
    public sendCommand(_commandType: string, _content?: string): void {
        // UT helper
    }

    public onCommand(_listener: (commandType: string, content: string) => void): void {
        // UT helper
    }

    public onError(_listener: (error: Error) => void): void {
        // UT helper
    }
}
