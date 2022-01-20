// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert';
import fs from 'fs';
import path from 'path';

import fetch from 'node-fetch';
//import rewire from 'rewire';

import { setExperimentStartupInfo, ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { RestServer, UnitTestHelpers } from 'rest_server';
import * as mock_netron_server from './mock_netron_server';

let restServer: RestServer;
let endPoint: string;
let endPointWithoutPrefix: string;
let netronHost: string;

const logFileContent = fs.readFileSync(path.join(__dirname, 'log/mock.log'));
const webuiIndexContent = fs.readFileSync(path.join(__dirname, 'static/index.html'));
const webuiScriptContent = fs.readFileSync(path.join(__dirname, 'static/script.js'));

before(async () => {
    await configRestServer();

    const netronPort = await mock_netron_server.start();
    netronHost = `localhost:${netronPort}`;
    UnitTestHelpers.setNetronUrl('http://' + netronHost);
});

async function configRestServer(urlPrefix?: string) {
    if (restServer !== undefined) {
        await restServer.shutdown();
    }

    // Set port, URL prefix, and log path.
    // There should be a better way to do this.
    // Maybe rewire? I can't get it work with TypeScript.
    setExperimentStartupInfo(
        true,
        path.basename(__dirname),  // hacking getLogDir()
        0,  // ask for a random port
        'local',
        path.dirname(__dirname),
        undefined,
        undefined,
        undefined,
        urlPrefix
    );

    UnitTestHelpers.setWebuiPath(path.join(__dirname, 'static'));

    restServer = new RestServer();
    await restServer.start();
    const port = UnitTestHelpers.getPort(restServer);

    endPointWithoutPrefix = `http://localhost:${port}`;
    endPoint = urlJoin(endPointWithoutPrefix, urlPrefix ?? '');
}

after(async () => {
    await restServer.shutdown();
});

function urlJoin(part1: string, part2: string): string {
    if (part1.endsWith('/')) {
        part1 = part1.slice(0, -1);
    }
    if (part2.startsWith('/')) {
        part2 = part2.slice(1);
    }
    if (part2 === '') {
        return part1;
    }
    return part1 + '/' + part2;
}

// NNI manager APIs are covered by old tests. In future RestServer should not be responsible for API implementation.

async function testLogs(): Promise<void> {
    const res = await fetch(urlJoin(endPoint, '/logs/mock.log'));
    const contentType = res.headers.get('Content-Type')!;
    assert.ok(res.ok);
    assert.ok(contentType.startsWith('text/plain'));  // content type can influence browser behavior
    assert.equal(await res.text(), logFileContent);
}

async function testNetronGet(): Promise<void> {
    const res = await fetch(urlJoin(endPoint, '/netron/mock/get-path'));
    const req = await res.json();  // the mock server send request info back as response
    assert.ok(res.ok);
    assert.equal(req.headers.host, netronHost);
    assert.equal(req.url, '/mock/get-path');
}

async function testNetronPost(): Promise<void> {
    const postData = 'hello netron';
    const res = await fetch(urlJoin(endPoint, '/netron/post-path'), { method: 'POST', body: postData });
    const req = await res.json();
    assert.ok(res.ok);
    assert.equal(req.url, '/post-path');
    assert.equal(req.body, postData);
}

async function testWebuiIndex(): Promise<void> {
    const res = await fetch(endPoint);
    assert.ok(res.ok);
    assert.equal(await res.text(), webuiIndexContent);
}

async function testWebuiResource(): Promise<void> {
    const res = await fetch(urlJoin(endPoint, '/script.js'));
    assert.ok(res.ok);
    assert.equal(await res.text(), webuiScriptContent);
}

async function testWebuiRouting(): Promise<void> {
    const res = await fetch(urlJoin(endPoint, '/not-exist'));
    assert.ok(res.ok);
    assert.equal(await res.text(), webuiIndexContent);
}

async function testOutsidePrefix(): Promise<void> {
    const res = await fetch(endPointWithoutPrefix);
    assert.equal(res.status, 404);
}

describe('## rest_server ##', () => {
    it('logs', () => testLogs());
    it('netron get', () => testNetronGet());
    it('netron post', () => testNetronPost());
    it('webui index', () => testWebuiIndex());
    it('webui resource', () => testWebuiResource());
    it('webui routing', () => testWebuiRouting());

    it('// re-configure rest server', () => configRestServer('url/prefix'));

    it('prefix logs', () => testLogs());
    it('prefix netron get', () => testNetronGet());
    it('prefix netron post', () => testNetronPost());
    it('prefix webui index', () => testWebuiIndex());
    it('prefix webui resource', () => testWebuiResource());
    it('prefix webui routing', () => testWebuiRouting());
    it('outside prefix', () => testOutsidePrefix());
});
