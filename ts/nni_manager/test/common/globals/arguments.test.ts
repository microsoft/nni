// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';

import { parseArgs } from 'common/globals/arguments';

const cmd = '--port 80 --experiment-id ID --action resume --experiments-directory DIR --log-level error';
const expected = {
    port: 80,
    experimentId: 'ID',
    action: 'resume',
    experimentsDirectory: 'DIR',
    logLevel: 'error',
    foreground: false,
    urlPrefix: '',

    mode: '',
    dispatcherPipe: undefined,
};

function testGoodShort(): void {
    const args = parseArgs(cmd.split(' '));
    assert.deepEqual(args, expected);
}

function testGoodLong(): void {
    const str = cmd + ' --url-prefix URL/prefix --foreground true';
    const args = parseArgs(str.split(' '));
    const expected = Object.assign({}, expected);
    expected.urlPrefix = 'URL/prefix';
    expected.foreground = true;
    assert.deepEqual(args, expected);
}

function testBadKey(): void {
    const str = cmd + ' --bad 1';
    assert.throws(() => parseArgs(str.split(' ')));
}

function testBadPos(): void {
    const str = cmd.replace('--port', 'port');
    assert.throws(() => parseArgs(str.split(' ')));
}

function testBadNum(): void {
    const str = cmd.replace('80', '8o');
    assert.throws(() => parseArgs(str.split(' ')));
}

function testBadBool(): void {
    const str = cmd + ' --foreground 1';
    assert.throws(() => parseArgs(str.split(' ')));
}

function testBadChoice(): void {
    const str = cmd.replace('resume', 'new');
    assert.throws(() => parseArgs(str.split(' ')));
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
