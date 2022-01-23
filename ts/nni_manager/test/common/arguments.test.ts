// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';

import { parseArgs } from 'common/globals/arguments';

const argsStr = '--port 80 --experiment-id ID --action resume --experiments-directory DIR --log-level error';
const argsExpected = {
    port: 80,
    experimentId: 'ID',
    action: 'resume',
    experimentsDirectory: 'DIR',
    logLevel: 'error',
    foreground: false,
    urlPrefix: '',

    mode: undefined,
    dispatcherPipe: undefined,
};

function testGoodShort(): void {
    const args = parseArgs(argsStr.split(' '));
    assert.deepEqual(args, argsExpected);
}

function testGoodLong(): void {
    const str = argsStr + ' --url-prefix URL/prefix --foreground true';
    const args = parseArgs(str.split(' '));
    const expected = Object.assign({ }, argsExpected);
    expected.urlPrefix = 'URL/prefix';
    expected.foreground = true;
    assert.deepEqual(args, expected);
}

function testBadKey(): void {
    const str = argsStr + ' --bad 1';
    assert.throws(() => parseArgs(str.split(' ')));
}

function testBadPos(): void {
    const str = argsStr.replace('--port', 'port');
    assert.throws(() => parseArgs(str.split(' ')));
}

function testBadNum(): void {
    const str = argsStr.replace('80', '8o');
    assert.throws(() => parseArgs(str.split(' ')));
}

function testBadBool(): void {
    const str = argsStr + ' --foreground 1';
    assert.throws(() => parseArgs(str.split(' ')));
}

function testBadChoice(): void {
    const str = argsStr.replace('resume', 'new');
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
