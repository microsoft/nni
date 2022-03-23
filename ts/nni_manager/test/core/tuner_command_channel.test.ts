// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import { setTimeout } from 'timers/promises';

import WebSocket from 'ws';

import { getWebSocketChannel, serveWebSocket } from 'core/tuner_command_channel';
import { UnitTestHelpers } from 'core/tuner_command_channel/web_socket_channel';

UnitTestHelpers.setHeartbeatInterval(10);  // for testError, must be set before serveWebSocket()

/** Test 1: start serving and let a client connect **/
async function testInit(): Promise<void> {
    server.on('connection', serveWebSocket);
    startClient();
    await getWebSocketChannel().init();
}

/** Test 2: send commands from server to client **/
async function testSend(): Promise<void> {
    const channel = getWebSocketChannel();

    channel.sendCommand(command1);
    channel.sendCommand(command2);
    await setTimeout(10);

    assert.equal(clientReceived.length, 2);
    assert.equal(clientReceived[0], command1);
    assert.equal(clientReceived[1], command2);
}

/** Test 3: send commands from client to server **/
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

/** Test 4: simulate client side crash */
async function testError(): Promise<void> {
    const channel = getWebSocketChannel();
    channel.onError(error => { catchedError = error; });

    // we have set heartbeat interval to 10ms, so pause for 50ms should make it timeout
    client.pause();
    await setTimeout(50);

    assert.notEqual(catchedError, undefined);
    client.resume();
}

describe('## tuner_command_channel ##', () => {
    it('init', async function () {
        this.timeout(1000);  // if it takes more than 1s, likely something wrong with the connection
        await testInit();
    });
    it('send', testSend);
    it('receive', testReceive);
    it('catch error', testError);
});

after(() => {
    client.close();
    server.close();
});

/** utilities **/

const command1 = 'T_hello';
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
