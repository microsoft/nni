// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import fs from 'fs';
import path from 'path';
import { setTimeout } from 'timers/promises';

import { LogStream, initLogStream } from 'common/globals/log_stream';
import globals from 'common/globals/unittest';

const lines = [ 'hello', '你好' ];
let logStream: LogStream;
const consoleLoggedLines: string[] = []

async function testWrite(): Promise<void> {
    logStream.writeLine(lines[0]);
    logStream.writeLine(lines[1]);
    await setTimeout(10);
    expectLineNumbers([0, 1]);
}

async function testClose(): Promise<void> {
    logStream.writeLine(lines[1]);
    logStream.writeLine(lines[0]);
    await logStream.close();
    expectLineNumbers([0, 1, 1, 0]);
}

// Assert specified lines are logged to correct place.
function expectLineNumbers(lineNumbers: number[]) {
    const expectConsole = globals.args.foreground ? lineNumbers.map(n => lines[n]) : [];
    const expectFile = lineNumbers.map(n => lines[n]).join('\n') + '\n';

    const file = fs.readFileSync(globals.paths.nniManagerLog, { encoding: 'utf8' });

    assert.deepEqual(consoleLoggedLines.join('\n'), expectConsole.join('\n'));
    assert.equal(file, expectFile);
}

describe('## globals.log_stream ##', () => {
    it('background', () => testWrite());
    it('background close', () => testClose());

    it('// switch to foreground', () => { switchForeground(); });

    it('foreground', () => testWrite());
    it('foreground close', () => testClose());
});

const origConsoleLog = console.log;
const tempDir = fs.mkdtempSync('nni-ut-');

before(() => {
    console.log = (line => { consoleLoggedLines.push(line); });
    globals.paths.nniManagerLog = path.join(tempDir, 'nnimanager.log');
    globals.args.foreground = false;
    logStream = initLogStream(globals.args, globals.paths);
});

function switchForeground() {
    logStream.close();
    consoleLoggedLines.length = 0;
    fs.rmSync(globals.paths.nniManagerLog);

    globals.args.foreground = true;
    logStream = initLogStream(globals.args, globals.paths);
}

after(() => {
    console.log = origConsoleLog;
    fs.rmSync(tempDir, { force: true, recursive: true });
    globals.reset();
});
