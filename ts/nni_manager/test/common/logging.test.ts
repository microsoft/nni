// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';

import globals from 'common/globals/unittest';
import { Logger, getLogger, getRobustLogger } from 'common/log';

/* test cases */

// Write a log message in different format for each level.
// Checks the log stream contains all messages.
function testDebugLevel() {
    stream.reset();
    globals.args.logLevel = 'debug';

    writeLogs1(getLogger('DebugLogger'));
    writeLogs2(getLogger('DebugLogger'));

    assert.match(stream.content, /DebugLogger/);

    assert.match(stream.content, /debug-message/);
    assert.match(stream.content, /info-message/);
    assert.match(stream.content, /warning-message/);
    assert.match(stream.content, /error-message/);
    assert.match(stream.content, /critical-message/);

    assert.equal(stderr, '');
}

// Write a log message in different format for each level.
// Check logs below specified log level are filtered.
function testWarningLevel() {
    stream.reset();
    globals.args.logLevel = 'warning';

    writeLogs1(getLogger('WarningLogger1'));
    writeLogs2(getLogger('WarningLogger2'));

    assert.match(stream.content, /WarningLogger1/);
    assert.match(stream.content, /WarningLogger2/);

    assert.doesNotMatch(stream.content, /debug-message/);
    assert.doesNotMatch(stream.content, /info-message/);
    assert.match(stream.content, /warning-message/);
    assert.match(stream.content, /error-message/);
    assert.match(stream.content, /critical-message/);

    assert.equal(stderr, '');
}

// Write some logs; simulate an error in log stream; then write other logs.
// Check logs after the error are written to stderr.
function testRobust() {
    stream.reset();
    globals.args.logLevel = 'info';

    const logger = getRobustLogger('RobustLogger');
    writeLogs1(logger);
    stream.error = true;
    writeLogs2(logger);

    assert.match(stream.content, /RobustLogger/);
    assert.doesNotMatch(stream.content, /debug-message/);
    assert.match(stream.content, /info-message/);
    assert.match(stream.content, /warning-message/);

    assert.match(stderr, /stream-error/);
    assert.match(stderr, /error-message/);
    assert.match(stderr, /critical-message/);
}

/* register */
describe('## logging ##', () => {
    before(beforeHook);

    it('low log level', testDebugLevel);
    it('high log level', testWarningLevel);
    it('robust', testRobust);

    after(afterHook);
});

/* helpers */

function writeLogs1(logger: Logger) {
    logger.debug('debug-message');
    logger.info(1, '2', 'info-message', 3);
    logger.warning(undefined, [null, 'warning-message']);
}

function writeLogs2(logger: Logger) {
    const recursiveObject: any = { 'message': 'error-message' };
    recursiveObject.recursive = recursiveObject;
    logger.error(recursiveObject);
    logger.critical(new Error('critical-message'));
}

class TestLogStream {
    public content: string = '';
    public error: boolean = false;

    reset(): void {
        this.content = '';
        this.error = false;
    }

    writeLine(line: string): void {
        if (this.error) {
            throw new Error('stream-error');
        }
        this.content += line + '\n';
    }

    writeLineSync(line: string): void {
        if (this.error) {
            throw new Error('stream-error');
        }
        this.content += line + '\n';
    }

    async close(): Promise<void> {
        /* empty */
    }
}

/* environment */

const stream = new TestLogStream();
const origConsoleError = console.error;
let stderr: string = '';

async function beforeHook() {
    globals.logStream = stream;
    console.error = (...args: any[]) => { stderr += args.join(' ') + '\n'; };
}

async function afterHook() {
    globals.reset();
    console.error = origConsoleError;
}
