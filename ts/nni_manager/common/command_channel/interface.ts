// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

export interface Command {
    type: string;
}
 
/**
 *  A command channel server serves one or more command channels.
 *  Each channel is connected to a client.
 *
 *  Normally each client has a unique channel URL,
 *  which can be got with `server.getChannelUrl(id)`.
 *
 *  The APIs might be changed to return `Promise<void>` in future.
 **/
export interface CommandChannelServer {
    // constructor(name: string, urlPath: string)
    start(): Promise<void>;
    shutdown(): Promise<void>;
    getChannelUrl(channelId: string): string;
    send(channelId: string, command: Command): void;
    onReceive(callback: (channelId: string, command: Command) => void): void;
}
