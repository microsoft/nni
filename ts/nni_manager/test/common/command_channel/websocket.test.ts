// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import { setTimeout } from 'timers/promises';

import WebSocket from 'ws';

import { WebSocketChannelServer, UnitTestHelpers } from 'common/command_channel/websocket';
import { DefaultMap } from 'common/default_map';
import { Deferred } from 'common/deferred';
import globals from 'common/globals/unittest';
import type { Command } from 'common/command_channel/interface';
import { RestServer, UnitTestHelpers as RestServerHelpers } from 'rest_server';

describe('## websocket command channel ##', () => {
    before(beforeHook);

    it('start', () => testStart());

    /* send and receive messages on normal connections */
    it('connect', () => testConnect());
    it('message', () => testMessage());

    /* client 1 proactively reconnect */
    it('reconnect', () => testReconnect());
    it('message', () => testMessage());

    /* client 2 loses responsive */
    it('no response', () => testNoResponse());
    it('message', () => testMessage());

    it('shutdown', () => testShutdown());

    after(afterHook);
});

/* global states */

const heartbeatInterval: number = 10;  // NOTE: increase this if the pipeline fails randomly
UnitTestHelpers.setHeartbeatInterval(heartbeatInterval);

let server!: WebSocketChannelServer;
let client1!: Client;
let client2!: Client;
const serverReceivedCommands: DefaultMap<string, Command[]> = new DefaultMap(Array) as any;

let restServer!: RestServer;

/* test cases */

async function testStart(): Promise<void> {
    server = new WebSocketChannelServer('ut', 'ut', 2);
    server.onReceive((channelId, command) => {
        serverReceivedCommands.get(channelId).push(unpackCommand(command));
    });
    await server.start();
    assert.equal(server.getChannelUrl('1'), `ws://localhost:${globals.args.port}/ut/1`);
}

async function testConnect(): Promise<void> {
    client1 = new Client(server.getChannelUrl('1'));
    client2 = new Client(server.getChannelUrl('2'));
    await Promise.all([ client1.opened.promise, client2.opened.promise ]);
}

async function testReconnect(): Promise<void> {
    const oldClient = client1;
    client1 = new Client(server.getChannelUrl('1'));
    await Promise.all([ oldClient.closed.promise, client1.opened.promise ]);
}

async function testNoResponse(): Promise<void> {
    await client2.mockNoResponse(heartbeatInterval * 5);
    client2 = new Client(server.getChannelUrl('2'));
    await client2.opened.promise;
}

async function testMessage(): Promise<void> {
    serverReceivedCommands.forEach(commands => { commands.length = 0; });
    client1.received.length = 0;
    client2.received.length = 0;

    server.send('1', packCommand(1));       // server -> 1
    client2.send(packCommand('二'));        // server <- 2
    client2.send(packCommand(3));           // server <- 2
    server.send('2', packCommand('四'));    // server -> 2
    client1.send(packCommand(5));           // server <- 1
    server.send('1', packCommand(6));       // server -> 1

    await setTimeout(heartbeatInterval);

    assert.deepEqual(client1.received, [ 1, 6 ]);
    assert.deepEqual(client2.received, [ '四' ]);
    assert.deepEqual(serverReceivedCommands.get('1'), [ 5 ]);
    assert.deepEqual(serverReceivedCommands.get('2'), [ '二', 3 ]);
}

async function testShutdown(): Promise<void> {
    await server.shutdown();
    await Promise.all([ client1.closed.promise, client2.closed.promise ]);
}

/* helpers */

async function beforeHook(): Promise<void> {
    globals.reset();
    //globals.showLog();
    restServer = new RestServer(0, '');
    await restServer.start();
    globals.args.port = RestServerHelpers.getPort(restServer);
}

async function afterHook() {
    if (restServer) {
        await restServer.shutdown();
    }
    globals.reset();
    RestServerHelpers.reset();
}

function packCommand(value: any): Command {
    return { type: 'ut', value } as Command;
}

function unpackCommand(command: Command): any {
    assert.equal(command.type, 'ut');
    return (command as any).value;
}

class Client {
    ws: WebSocket;
    received: any[] = [];
    opened: Deferred<void> = new Deferred();
    closed: Deferred<void> = new Deferred();

    constructor(url: string) {
        this.ws = new WebSocket(url);
        this.ws.on('message', (data, _isBinary) => {
            const command = JSON.parse(data.toString());
            this.received.push(unpackCommand(command));
        });
        this.ws.on('open', () => {
            this.opened.resolve();
        });
        this.ws.on('close', () => {
            this.closed.resolve();
        });
    }

    send(command: Command): void {
        this.ws.send(JSON.stringify(command));
    }

    async mockNoResponse(time: number): Promise<void> {
        this.ws.pause();
        await setTimeout(time);
        this.ws.terminate();
        this.ws.resume();
    }
}
