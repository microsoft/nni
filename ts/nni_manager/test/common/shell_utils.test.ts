// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import fs from 'fs';
import fsPromises from 'fs/promises';
import os from 'os';
import path from 'path';

import { createScriptFile } from 'common/shellUtils';

let tempDir: string | null = null;
const asciiScript = 'echo hello\necho world';
const unicodeScript = 'echo 你好\necho 世界';

/* test cases */

async function testAscii(): Promise<void> {
    const file = path.join(tempDir!, 'test1.ps1');
    await createScriptFile(file, asciiScript);
    const script = await fsPromises.readFile(file, { encoding: 'utf8' });
    assert.equal(script, asciiScript);
}

async function testUnicode(): Promise<void> {
    const file = path.join(tempDir!, 'test2.ps1');
    await createScriptFile(file, unicodeScript);
    const script = await fsPromises.readFile(file, { encoding: 'utf8' });
    assert.equal(script, '\uFEFF' + unicodeScript);
}

async function testBash(): Promise<void> {
    const file = path.join(tempDir!, 'test.sh');
    await createScriptFile(file, unicodeScript);
    const script = await fsPromises.readFile(file, { encoding: 'utf8' });
    assert.equal(script, unicodeScript);
}

/* environment */

function beforeHook() {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nni-ut-'));
}

function afterHook() {
    if (tempDir) {
        fs.rmSync(tempDir, { force: true, recursive: true });
    }
}

/* register */

describe('## common.shell_utils ##', () => {
    before(beforeHook);

    it('powershell ascii', () => testAscii());
    it('powershell unicode', () => testUnicode());
    it('bash unicode', () => testBash());

    after(afterHook);
});
