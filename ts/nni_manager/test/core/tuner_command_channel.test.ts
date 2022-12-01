// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import { setTimeout } from 'timers/promises';

import WebSocket from 'ws';

import { Deferred } from 'common/deferred';
import { getWebSocketChannel, serveWebSocket } from 'core/tuner_command_channel';
import { UnitTestHelpers } from 'core/tuner_command_channel/websocket_channel';

const heartbeatInterval: number = 10;

// for testError, must be set before serveWebSocket()
UnitTestHelpers.setHeartbeatInterval(heartbeatInterval);

/* test cases */

// Start serving and let a client connect.
async function testInit(): Promise<void> {
    const channel = getWebSocketChannel();
    channel.onCommand(command => { serverReceived.push(command); });
    channel.onError(error => { catchedError = error; });

    server.on('connection', serveWebSocket);
    client1 = new Client('client1');
    await channel.init();
}

// Send commands from server to client.
async function testSend(client: Client): Promise<void> {
    const channel = getWebSocketChannel();

    channel.sendCommand(command1);
    channel.sendCommand(command2);
    await setTimeout(heartbeatInterval);

    assert.deepEqual(client.received, [command1, command2]);
}

// Send commands from client to server.
async function testReceive(client: Client): Promise<void> {
    serverReceived.length = 0;

    client.ws.send(command2);
    client.ws.send(command1);
    await setTimeout(heartbeatInterval);

    assert.deepEqual(serverReceived, [command2, command1]);
}

// Simulate client side crash.
async function testError(): Promise<void> {
    if (process.platform !== 'linux') {
        // it is performance sensitive for the test case to yield error,
        // but windows & mac agents of devops are too slow
        client1.ws.terminate();
        return;
    }

    // we have set heartbeat interval to 10ms, so pause for 30ms should make it timeout
    client1.ws.pause();
    await setTimeout(heartbeatInterval * 3);
    client1.ws.resume();

    assert.notEqual(catchedError, undefined);
}

// If the client losses connection by accident but not crashed, it will reconnect.
async function testReconnect(): Promise<void> {
    client2 = new Client('client2');
    await client2.deferred.promise;
}

// Clean up.
async function testShutdown(): Promise<void> {
    const channel = getWebSocketChannel();
    await channel.shutdown();

    client1.ws.close();
    client2.ws.close();
    server.close();
}

/* register */
describe('## tuner_command_channel ##', () => {
    it('init', testInit);

    it('send', () => testSend(client1));
    it('receive', () => testReceive(client1));

    it('mock timeout', testError);
    it('reconnect', testReconnect);

    it('send after reconnect', () => testSend(client2));
    it('receive after reconnect', () => testReceive(client2));

    it('shutdown', testShutdown);
});

/** helpers **/

const command1 = 'T_hello world';
const command2 = 'T_你好';

const server = new WebSocket.Server({ port: 0 });
let client1!: Client;
let client2!: Client;

const serverReceived: string[] = [];
let catchedError: Error | undefined;

class Client {
    name: string;
    received: string[] = [];
    ws!: WebSocket;
    deferred: Deferred<void> = new Deferred();

    constructor(name: string) {
        this.name = name;
        const port = (server.address() as any).port;
        this.ws = new WebSocket(`ws://localhost:${port}`);
        this.ws.on('message', (data, _isBinary) => {
            this.received.push(data.toString());
        });
        this.ws.on('open', () => {
            this.deferred.resolve();
        });
    }
}
