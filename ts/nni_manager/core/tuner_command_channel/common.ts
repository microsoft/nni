// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { BaseCommand } from "core/semanticCommand";

export interface IpcInterface {
    init(): Promise<void>;
    sendCommand(command: BaseCommand): void;
    onCommand(listener: (commandType: string, content: string) => void): void;
    onError(listener: (error: Error) => void): void;
}
