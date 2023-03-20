// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Common interface of command channels.
 *
 *  A command channel is a duplex connection which supports sending and receiving JSON commands.
 *
 *  Typically a command channel implementation consists of a server and a client.
 *
 *  The server should listen to a URL prefix like `http://localhost:8080/example/`;
 *  and each client should connect to a unique URL containing the prefix, e.g. `http://localhost:8080/example/channel1`.
 *  The client's URL should be created with `server.getChannelUrl(channelId, serverIp)`.
 *
 *  We currently have implemented one full feature command channel, the WebSocket channel,
 *  and a simplified one, the HTTP channel.
 *  In v3.1 release we might implement file command channel and AzureML command channel.
 *
 *  The clients might have a Python version locates in `nni/runtime/command_channel`.
 *  The TypeScript and Python version should be interchangable.
 **/

/**
 *  A command is a JSON object.
 *
 *  The object has only one mandatory entry, `type`.
 *
 *  The type string should not be surrounded by underscore (e.g. `_nop_`),
 *  unless you are dealing with the underlying implementation of a specific command channel;
 *  it should never starts with two underscores (e.g. `__command`) in any circumstance.
 **/
export type Command = any;

// Maybe it's better to disable `noPropertyAccessFromIndexSignature` in tscofnig?
//  export interface Command {
//      type: string;
//      [key: string]: any;
//  }
 
/**
 *  `CommandChannel` is the base interface used by both the servers and the clients.
 *
 *  For servers, channels can be got with `onConnection()` event listener.
 *  For clients, a channel can be created with the client subclass' constructor.
 *
 *  The channel should be fault tolerant to some extend. It has three different types of closing related events:
 *
 *   1. Close: The channel is intentionally closed.
 *
 *   2. Lost: The channel is temporarily unavailable and is trying to recover.
 *      The user of this class should examine the peer's status out-of-band when receiving "lost" event.
 *
 *   3. Error: The channel is dead and cannot recover.
 *      A "close" event may or may not occur following this event. Do not rely on that.
 **/
export interface CommandChannel {
    readonly name: string;  // for better logging

    enableHeartbeat(intervalMilliseconds?: number): void;

    /**
     *  Graceful (intentional) close.
     *  A "close" event will be emitted by `this` and the peer.
     **/
    close(reason: string): void;

    /**
     *  Force close. Should only be used when the channel is not working.
     *  An "error" event may be emitted by `this`.
     *  A "lost" and/or "error" event will be emitted by the peer, if its process is still alive.
     **/
    terminate(reason: string): void;

    send(command: Command): void;

    /**
     *  The async version should try to ensures the command is successfully sent to the peer.
     *  But this is not guaranteed.
     **/
    sendAsync(command: Command): Promise<void>;

    onReceive(callback: (command: Command) => void): void;
    onCommand(commandType: string, callback: (command: Command) => void): void;

    onClose(callback: (reason?: string) => void): void;
    onError(callback: (error: Error) => void): void;
    onLost(callback: () => void): void;
}

/**
 *  Client side of a command channel.
 *
 *  The constructor should have no side effects.
 *
 *  The listeners should be registered before calling `connect()`,
 *  or the first few commands might be missed.
 *
 *  Example usage:
 *
 *      const client = new WsChannelClient('example', 'ws://1.2.3.4:8080/server/channel_id');
 *      await client.connect();
 *      client.send(command);
 **/
export interface CommandChannelClient extends CommandChannel {
    // constructor(name: string, url: string);

    connect(): Promise<void>;

    /**
     *  Typically an alias of `close()`.
     **/
    disconnect(reason?: string): Promise<void>;
}

/**
 *  Server side of a command channel.
 *
 *  The consructor should have no side effects.
 *
 *  The listeners should be registered before calling `start()`.
 *
 *  Example usage:
 *
 *      const server = new WsChannelServer('example_server', '/server_prefix');
 *      const url = server.getChannelUrl('channel_id');
 *      const client = new WsChannelClient('example_client', url);
 *      await server.start();
 *      await client.connect();
 *
 *  There two ways to listen to command:
 *
 *   1. Handle all clients' commands in one space:
 *
 *          server.onReceive((channelId, command) => { ... });
 *          server.send(channelId, command);
 *
 *   2. Maintain a `WsChannel` instance for each client:
 *
 *          server.onConnection((channelId, channel) => {
 *              channel.onCommand(command => { ... });
 *              channel.send(command);
 *          });
 **/
export interface CommandChannelServer {
    // constructor(name: string, urlPath: string);

    start(): Promise<void>;
    shutdown(): Promise<void>;

    /**
     *  When `ip` is missing, it should default to localhost.
     **/
    getChannelUrl(channelId: string, ip?: string): string;

    send(channelId: string, command: Command): void;
    onReceive(callback: (channelId: string, command: Command) => void): void;
    onConnection(callback: (channelId: string, channel: CommandChannel) => void): void;
}
