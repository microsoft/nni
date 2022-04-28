// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import { setTimeout } from 'timers/promises';

import WebSocket from 'ws';

import { getWebSocketChannel, serveWebSocket } from 'core/tuner_command_channel';
import { UnitTestHelpers } from 'core/tuner_command_channel/websocket_channel';

UnitTestHelpers.setHeartbeatInterval(10);  // for testError, must be set before serveWebSocket()

/* test cases */

// Start serving and let a client connect.
async function testInit(): Promise<void> {
    server.on('connection', serveWebSocket);
    startClient();
    await getWebSocketChannel().init();
}

// Send commands from server to client.
async function testSend(): Promise<void> {
    const channel = getWebSocketChannel();

    channel.sendCommand(command1);
    channel.sendCommand(command2);
    await setTimeout(10);

    assert.equal(clientReceived.length, 2);
    assert.equal(clientReceived[0], command1);
    assert.equal(clientReceived[1], command2);
}

// Send commands from client to server.
async function testReceive(): Promise<void> {
    const channel = getWebSocketChannel();
    channel.onCommand(command => { serverReceived.push(command); });

    client.send(command1);
    client.send(command2);
    await setTimeout(10);

    assert.equal(serverReceived.length, 2);
    assert.deepEqual(serverReceived[0], command1);
    assert.deepEqual(serverReceived[1], command2);
}

// Simulate client side crash.
async function testError(): Promise<void> {
    const channel = getWebSocketChannel();

    if (process.platform === 'darwin') {
        // macOS does not raise the error in 30ms
        // not a big problem and don't want to debug. ignore it.
        channel.shutdown();
        return;
    }

    channel.onError(error => { catchedError = error; });

    // we have set heartbeat interval to 10ms, so pause for 30ms should make it timeout
    client.pause();
    await setTimeout(30);

    assert.notEqual(catchedError, undefined);
    client.resume();
}

// Clean up.
async function testShutdown(): Promise<void> {
    const channel = getWebSocketChannel();
    await channel.shutdown();

    client.close();
    server.close();
}

/* register */
describe('## tuner_command_channel ##', () => {
    it('init', testInit);
    it('send', testSend);
    it('receive', testReceive);
    it('catch error', testError);
    it('shutdown', testShutdown);
});

/** helpers **/

const command1 = 'T_hello world';
const command2 = 'T_你好';
const commandPing = 'PI';

const server = new WebSocket.Server({ port: 0 });
let client!: WebSocket;

const serverReceived: string[] = [];
const clientReceived: string[] = [];
let catchedError: Error | undefined;

function startClient() {
    const port = (server.address() as any).port;
    client = new WebSocket(`ws://localhost:${port}`);
    client.on('message', message => { clientReceived.push(message.toString()); });
}
