// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

export interface IpcInterface {
    sendCommand(commandType: string, content?: string): void;
    onCommand(listener: (commandType: string, content: string) => void): void;
    onError(listener: (error: Error) => void): void;
}
