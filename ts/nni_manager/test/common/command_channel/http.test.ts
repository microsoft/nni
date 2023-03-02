// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import { setTimeout } from 'timers/promises';

import globals from 'common/globals/unittest';
import { HttpChannelServer, UnitTestHelpers } from 'common/command_channel/http';
import { Command } from 'common/command_channel/interface';
import { RestServer, UnitTestHelpers as RestServerHelpers } from 'rest_server';

let server: HttpChannelServer;
const serverReceivedCommands: Record<string, Command[]> = { '1': [], '2': [] };

let restServer: RestServer;
let port: number;

const asciiCommand = { type: 'hello', message: 'world' };
const unicodeCommand = { type: '你好', content: '世界' };

describe('## http command channel ##', () => {
    before(beforeHook);

    it('start', () => testStart());
    it('server <- client-1', () => testReceive('1'));
    it('server -> client-2 (before polling)', () => testSendBefore('2'));
    it('server <- client-2', () => testReceive('2'));
    it('server -> client-1 (short after polling)', () => testSendAfter('1', 10));
    it('server -> client-2 (long after polling)', () => testSendAfter('2', 30));
    it('stop', () => testStop());

    after(afterHook);
});

async function testStart(): Promise<void> {
    UnitTestHelpers.setTimeoutMilliseconds(20);
    server = new HttpChannelServer('ut', 'ut');
    server.onReceive((channelId, command) => {
        serverReceivedCommands[channelId].push(command);
    });
    await server.start();
    assert.equal(server.getChannelUrl('1'), `http://localhost:${port}/ut/1`);
}

async function testReceive(id: string): Promise<void> {
    await clientSend(id, asciiCommand);
    await clientSend(id, unicodeCommand);
    await setTimeout(10);
    assert.deepEqual(serverReceivedCommands[id], [ asciiCommand, unicodeCommand ]);
}

async function testSendBefore(id: string): Promise<void> {
    server.send(id, asciiCommand);
    const command = await clientReceive(id);
    assert.deepEqual(command, asciiCommand);
}

async function testSendAfter(id: string, delay: number): Promise<void> {
    const promise = clientReceive(id);
    await setTimeout(delay);
    server.send(id, unicodeCommand);
    const command = await promise;
    assert.deepEqual(command, unicodeCommand);
}

async function testStop(): Promise<void> {
    const promise = clientReceive('1');
    await setTimeout(10);
    await server.shutdown();
    const response = await promise;
    assert.equal(response, null);
}

/* Helper and environment */

async function clientSend(id: string, command: Command): Promise<void> {
    const url = server.getChannelUrl(id);
    await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(command),
    });
}

async function clientReceive(id: string): Promise<Command | null> {
    const url = server.getChannelUrl(id);
    for (let i = 0; i < 10; i++) {
        const r = await fetch(url);
        if (r.status === 200) {
            return r.json();
        }
        if (r.status === 408) {
            continue;
        }
        if (r.status === 410) {
            return null;
        }
        throw new Error(`Unexpected status ${r.status}`);
    }
    throw new Error('No command in 10s');
}

async function beforeHook(): Promise<void> {
    globals.reset();
    restServer = new RestServer(0, '');
    await restServer.start();
    port = RestServerHelpers.getPort(restServer);
    globals.args.port = port;
}

async function afterHook() {
    if (restServer) {
        await restServer.shutdown();
    }
    globals.reset();
    RestServerHelpers.reset();
}
