// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';

import { parseArgs } from 'common/globals/arguments';

const command = [
    '--port 80',
    '--experiment-id ID',
    '--action resume',
    '--experiments-directory DIR',
    '--log-level error',
    '--python-interpreter python',
].join(' ');

const expected = {
    port: 80,
    experimentId: 'ID',
    action: 'resume',
    experimentsDirectory: 'DIR',
    logLevel: 'error',
    foreground: false,
    urlPrefix: '',
    tunerCommandChannel: null,
    pythonInterpreter: 'python',

    mode: '',
};

function testGoodShort(): void {
    const args = parseArgs(command.split(' '));
    assert.deepEqual(args, expected);
}

function testGoodLong(): void {
    const cmd = command + ' --url-prefix URL/prefix --foreground true';
    const args = parseArgs(cmd.split(' '));
    const expectedLong = Object.assign({}, expected);
    expectedLong.urlPrefix = 'URL/prefix';
    expectedLong.foreground = true;
    assert.deepEqual(args, expectedLong);
}

function testBadKey(): void {
    const cmd = command + ' --bad 1';
    assert.throws(() => parseArgs(cmd.split(' ')));
}

function testBadPos(): void {
    const cmd = command.replace('--port', 'port');
    assert.throws(() => parseArgs(cmd.split(' ')));
}

function testBadNum(): void {
    const cmd = command.replace('80', '8o');
    assert.throws(() => parseArgs(cmd.split(' ')));
}

function testBadBool(): void {
    const cmd = command + ' --foreground 1';
    assert.throws(() => parseArgs(cmd.split(' ')));
}

function testBadChoice(): void {
    const cmd = command.replace('resume', 'new');
    assert.throws(() => parseArgs(cmd.split(' ')));
}

describe('## globals.arguments ##', () => {
    it('good short', () => testGoodShort());
    it('good long', () => testGoodLong());
    it('bad key arg', () => testBadKey());
    it('bad positional arg', () => testBadPos());
    it('bad number', () => testBadNum());
    it('bad boolean', () => testBadBool());
    it('bad choice', () => testBadChoice());
});
