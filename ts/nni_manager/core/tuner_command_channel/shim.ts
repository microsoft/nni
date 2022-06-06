// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { BaseCommand } from 'core/semanticCommand';
import type { IpcInterface } from './common';
import { WebSocketChannel, getWebSocketChannel } from './websocket_channel';

export async function createDispatcherInterface(): Promise<IpcInterface> {
    return new WsIpcInterface();
}

class WsIpcInterface implements IpcInterface {
    private channel: WebSocketChannel = getWebSocketChannel();

    public async init(): Promise<void> {
        await this.channel.init();
    }

    public sendCommand(command: BaseCommand): void {
        const legacyCommand = command.toLegacyCommand();
        const commandType = legacyCommand.slice(0, 2);
        if (commandType !== 'PI') {  // ping is handled with WebSocket protocol
            this.channel.sendCommand(legacyCommand);
            if (commandType === 'TE') {
                this.channel.shutdown();
            }
        }
    }

    public onCommand(listener: (commandType: string, content: string) => void): void {
        this.channel.onCommand((command: string) => {
            listener(command.slice(0, 2), command.slice(2));
        });
    }

    public onError(listener: (error: Error) => void): void {
        this.channel.onError(listener);
    }
}
