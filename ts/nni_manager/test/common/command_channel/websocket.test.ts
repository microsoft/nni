// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'node:assert/strict';
import { setTimeout } from 'node:timers/promises';

import type {
    Command, CommandChannel, CommandChannelClient, CommandChannelServer
} from 'common/command_channel/interface';
import { UnitTestHelper as Helper } from 'common/command_channel/websocket/channel';
import { WsChannelClient, WsChannelServer } from 'common/command_channel/websocket/index';
import { globals } from 'common/globals/unittest';
import { RestServerCore } from 'rest_server/core';

describe('## websocket command channel ##', () => {
    before(beforeHook);

    it('start', testServerStart);

    it('connect', testClientStart);
    it('message', testMessage);

    it('reconnect', testReconnect);
    it('message', testMessage);

    it('handle error', testError);
    it('shutdown', testShutdown);

    after(afterHook);
});

/* test cases */

async function testServerStart(): Promise<void> {
    ut.server = new WsChannelServer('ut_server', 'ut');

    ut.server.onReceive((channelId, command) => {
        if (channelId === '1') {
            ut.events.push({ event: 'server_receive_1', command });
        }
    });

    ut.server.onConnection((channelId, channel) => {
        ut.events.push({ event: 'connect', channelId, channel });

        channel.onClose(reason => {
            ut.events.push({ event: `client_close_${channelId}`, reason });
        });
        channel.onError(error => {
            ut.events.push({ event: `client_error_${channelId}`, error });
        });
        channel.onLost(() => {
            ut.events.push({ event: `client_lost_${channelId}` });
        });

        if (channelId === '1') {
            ut.serverChannel1 = channel;
        }

        if (channelId === '2') {
            ut.serverChannel2 = channel;
            channel.onReceive(command => {
                ut.events.push({ event: 'server_receive_2', command });
            });
        }
    });

    await ut.server.start();
}

async function testClientStart(): Promise<void> {
    const url1 = ut.server.getChannelUrl('1');
    const url2 = ut.server.getChannelUrl('2', '127.0.0.1');
    assert.equal(url1, `ws://localhost:${globals.args.port}/ut/1`);
    assert.equal(url2, `ws://127.0.0.1:${globals.args.port}/ut/2`);

    ut.client1 = new WsChannelClient('ut_client_1', url1);
    ut.client2 = new WsChannelClient('ut_client_2', url2);

    ut.client1.onReceive(command => {
        ut.events.push({ event: 'client_receive_1', command });
    });
    ut.client2.onCommand('ut_command', command => {
        ut.events.push({ event: 'client_receive_2', command });
    });

    ut.client2.onClose(reason => {
        ut.events.push({ event: 'server_close_2', reason });
    });

    await Promise.all([
        ut.client1.connect(),
        ut.client2.connect(),
    ]);

    assert.equal(ut.events[0].event, 'connect');
    assert.equal(ut.events[1].event, 'connect');
    assert.equal(Number(ut.events[0].channelId) + Number(ut.events[1].channelId), 3);
    assert.equal(ut.events.length, 2);

    ut.events.length = 0;
}

async function testReconnect(): Promise<void> {
    const ws = (ut.client1 as any).connection.ws;  // NOTE: private api
    ws.pause();
    await setTimeout(heartbeatTimeout);
    ws.terminate();
    ws.resume();

    // mac pipeline can be slow
    for (let i = 0; i < 10; i++) {
        await setTimeout(heartbeat);
        if (ut.events.length > 0) {
            break;
        }
    }

    assert.ok(ut.countEvents('client_lost_1') >= 1);
    assert.ok(ut.countEvents('client_close_1') == 0);
    assert.ok(ut.countEvents('client_error_1') == 0);
    assert.ok(ut.countEvents('connect') == 0);  // reconnect is not connect

    ut.events.length = 0;
}

async function testMessage(): Promise<void> {
    ut.server.send('1', ut.packCommand(1));
    await ut.client2.sendAsync(ut.packCommand(2));
    ut.client2.send(ut.packCommand('三'));
    ut.server.send('2', ut.packCommand('4'));
    ut.client1.send(ut.packCommand(5));
    ut.server.send('1', ut.packCommand(6));

    await setTimeout(heartbeat);

    assert.deepEqual(ut.filterCommands('client_receive_1'), [ 1, 6 ]);
    assert.deepEqual(ut.filterCommands('client_receive_2'), [ '4' ]);
    assert.deepEqual(ut.filterCommands('server_receive_1'), [ 5 ]);
    assert.deepEqual(ut.filterCommands('server_receive_2'), [ 2, '三' ]);

    ut.events.length = 0;
}

async function testError(): Promise<void> {
    ut.client2.terminate('client 2 terminate');
    await setTimeout(terminateTimeout * 1.1);

    assert.ok(ut.countEvents('client_close_2') == 0);
    assert.ok(ut.countEvents('client_error_2') == 1);

    ut.events.length = 0;
}

async function testShutdown(): Promise<void> {
    await ut.server.shutdown();

    assert.equal(ut.countEvents('client_close_1'), 1);

    ut.events.length = 0;
}

/* helpers and states */

// NOTE: Increase these numbers if it fails randomly
const heartbeat = 10;
const heartbeatTimeout = 50;
const terminateTimeout = 100;

async function beforeHook(): Promise<void> {
    globals.reset();

    ut.rest = new RestServerCore();
    await ut.rest.start();

    Helper.setHeartbeatInterval(heartbeat);
    Helper.setTerminateTimeout(terminateTimeout);
}

async function afterHook(): Promise<void> {
    Helper.reset();
    await ut.rest?.shutdown();
    globals.reset();
}

class UnitTestStates {
    server!: CommandChannelServer;
    client1!: CommandChannelClient;
    client2!: CommandChannelClient;
    serverChannel1!: CommandChannel;
    serverChannel2!: CommandChannel;
    events: any[] = [];

    rest!: RestServerCore;

    countEvents(event: string): number {
        return this.events.filter(e => (e.event === event)).length;
    }

    filterCommands(event: string): any[] {
        return this.events.filter(e => (e.event === event)).map(e => e.command.value);
    }

    packCommand(value: any): Command {
        return { type: 'ut_command', value };
    }
}

const ut = new UnitTestStates();
