// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//export interface Command {
//    type: string;
//    [key: string]: any;
//}
export type Command = any;
 
export interface CommandChannel {
    readonly name: string;

    enableHeartbeat(intervalMilliseconds?: number): void;
    close(reason: string): void;
    terminate(reason: string): void;

    send(command: Command): void;
    /**
     *  Async version of `send()` that (partially) ensures the command is successfully sent to peer.
     **/
    sendAsync(command: Command): Promise<void>;

    onReceive(callback: (command: Command) => void): void;
    onCommand(commandType: string, callback: (command: Command) => void): void;

    onClose(callback: (reason?: string) => void): void;
    onError(callback: (error: Error) => void): void;
    onLost(callback: () => void): void;
}

export interface CommandChannelClient extends CommandChannel {
    // constructor(url: string, name?: string);
    connect(): Promise<void>;
    disconnect(reason?: string): Promise<void>;
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
    // constructor(urlPath: string, name?: string);
    start(): Promise<void>;
    shutdown(): Promise<void>;
    getChannelUrl(channelId: string, ip?: string): string;
    send(channelId: string, command: Command): void;
    onReceive(callback: (channelId: string, command: Command) => void): void;
    onConnection(callback: (channelId: string, channel: CommandChannel) => void): void;
}
