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
let consoleContent: string = '';

/* test cases */

// Test cases will be run twice, the first time in background mode (write to log file only),
// and the second time in foreground mode (write to log file + stdout).

// Write 2 lines and wait 10 ms for it to flush.
async function testWrite(): Promise<void> {
    logStream.writeLine(lines[0]);
    logStream.writeLine(lines[1]);
    await setTimeout(10);

    const expected = [ lines[0], lines[1] ].join('\n') + '\n';
    const fileContent = fs.readFileSync(globals.paths.nniManagerLog, { encoding: 'utf8' });
    assert.equal(fileContent, expected);
    assert.equal(consoleContent, globals.args.foreground ? expected : '');
}

// Write 2 lines synchronously. It should not need to flush.
async function testWriteSync(): Promise<void> {
    logStream.writeLineSync(lines[0]);
    logStream.writeLineSync(lines[1]);

    const expected = [ lines[0], lines[1], lines[0], lines[1] ].join('\n') + '\n';
    const fileContent = fs.readFileSync(globals.paths.nniManagerLog, { encoding: 'utf8' });
    assert.equal(fileContent, expected);
    assert.equal(consoleContent, globals.args.foreground ? expected : '');
}

// Write 2 lines and close stream. It should guarantee to flush.
async function testClose(): Promise<void> {
    logStream.writeLine(lines[1]);
    logStream.writeLine(lines[0]);
    await logStream.close();

    const expected = [ lines[0], lines[1], lines[0], lines[1], lines[1], lines[0] ].join('\n') + '\n';
    const fileContent = fs.readFileSync(globals.paths.nniManagerLog, { encoding: 'utf8' });
    assert.equal(fileContent, expected);
    assert.equal(consoleContent, globals.args.foreground ? expected : '');
}

/* register test cases */

describe('## globals.log_stream ##', () => {
    before(beforeHook);

    it('background', () => testWrite());
    it('background sync', () => testWriteSync());
    it('background close', () => testClose());

    it('// switch to foreground', () => { switchForeground(); });

    it('foreground', () => testWrite());
    it('foreground sync', () => testWriteSync());
    it('foreground close', () => testClose());

    after(afterHook);
});

/* configure test environment */

const origConsoleLog = console.log;
const tempDir = fs.mkdtempSync('nni-ut-');

function beforeHook() {
    console.log = (line => { consoleContent += line + '\n'; });
    globals.paths.nniManagerLog = path.join(tempDir, 'nnimanager.log');
    globals.args.foreground = false;
    logStream = initLogStream(globals.args, globals.paths);
}

function switchForeground() {
    logStream.close();
    consoleContent = '';
    fs.rmSync(globals.paths.nniManagerLog);

    globals.args.foreground = true;
    logStream = initLogStream(globals.args, globals.paths);
}

function afterHook() {
    console.log = origConsoleLog;
    fs.rmSync(tempDir, { force: true, recursive: true });
    globals.reset();
}
