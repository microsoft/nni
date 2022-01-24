// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import fs from 'fs';
import path from 'path';

import { LogStream } from 'common/globals/log_stream';
import globals from 'common/globals/unittest';

const lines = [ 'hello', '你好', 'close', 'open' ];
let logStream: LogStream;
const consoleLoggedLines: string[] = []

// Assert specified lines are logged to correct place.
function expectLineNumbers(lineNumbers: number[]) {
    const expectConsole = globals.args.foreground ? lineNumbers.map(n => lines[n]) : [];
    const expectFile = lineNumbers.map(n => lines[n]).join('\n') + '\n';

    const file = fs.readFileSync(globals.paths.nniManagerLog, { encoding: 'utf8' });

    assert.deepEqual(consoleLoggedLines, expectConsole);
    assert.equal(file, expectFile);
}

function testEn() {
    logStream.writeLine(lines[0]);
    expectLineNumbers([0]);
}

function testZh() {
    logStream.writeLine(lines[1]);
    expectLineNumbers([0, 1]);
}

function testClose() {
    logStream.close();

    logStream.writeLine(lines[2]);
    expectLineNumbers([0, 1]);
}

function testOpen() {
    logStream.open();

    logStream.writeLine(lines[3]);
    expectLineNumbers([0, 1, 3]);
}

describe('## globals.log_stream ##', () => {
    it('background', () => testEn());
    it('background unicode', () => testZh());
    it('background close', () => testClose());
    it('background open', () => testOpen());

    it('// switch to foreground', () => { switchForeground(); });

    it('foreground', () => testEn());
    it('foreground unicode', () => testZh());
    it('foreground close', () => testClose());
    it('foreground open', () => testOpen());
});

const origConsoleLog = console.log;
const tempDir = fs.mkdtempSync('ut-');

before(() => {
    console.log = ((line) => { consoleLoggedLines.push(line); });
    globals.paths.nniManagerLog = path.join(tempDir, 'nnimanager.log');
    globals.args.foreground = false;
    logStream = new LogStream(globals.args, globals.paths);
});

function switchForeground() {
    logStream.close();
    consoleLoggedLines.length = 0;
    fs.rmSync(globals.paths.nniManagerLog);

    globals.args.foreground = true;
    logStream = new LogStream(globals.args, globals.paths);
}

after(() => {
    console.log = origConsoleLog;
    fs.rmSync(tempDir, { force: true, recursive: true });
});
