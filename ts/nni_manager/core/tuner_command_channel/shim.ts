// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import type { IpcInterface } from './common';
import { WebSocketChannel, getWebSocketChannel } from './websocket_channel';

export async function createDispatcherInterface(): Promise<IpcInterface> {
    return new WsIpcInterface();
}

class WsIpcInterface implements IpcInterface {
    private channel: WebSocketChannel = getWebSocketChannel();
    private commandListener?: (commandType: string, content: string) => void;
    private errorListener?: (error: Error) => void;

    constructor() {
        this.channel.onCommand((command: string) => {
            const commandType = command.slice(0, 2);
            const content = command.slice(2);
            if (commandType === 'ER') {
                if (this.errorListener !== undefined) {
                    this.errorListener(new Error(content));
                }
            } else {
                if (this.commandListener !== undefined) {
                    this.commandListener(commandType, content);
                }
            }
        });
    }

    public async init(): Promise<void> {
        await this.channel.init();
    }

    public sendCommand(commandType: string, content: string = ''): void {
        if (commandType !== 'PI') {  // ping is handled with WebSocket protocol
            this.channel.sendCommand(commandType + content);
            if (commandType === 'TE') {
                this.channel.shutdown();
            }
        }
    }

    public onCommand(listener: (commandType: string, content: string) => void): void {
        this.commandListener = listener;
    }

    public onError(listener: (error: Error) => void): void {
        this.errorListener = listener;
    }
}
