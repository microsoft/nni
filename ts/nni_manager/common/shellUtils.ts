// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fsPromises from 'fs/promises';

// for readability
const singleQuote = "'";
const doubleQuote = '"';
const backtick = '`';
const backslash = '\\';
const doubleBacktick = '``';
const doubleBackslash = '\\\\';
const newline = '\n';

/**
 *  Convert a string into quoted and escaped string for shell script.
 *  This function supports multi-line strings as well.
 *
 *  Examples:
 *      hello  -->  'hello'
 *      C:\Program Files\$app  -->  'C:\Program Files\$app'
 *      a'b"c$d<ENTER>e\f`g  -->  $'a\'b"c$d\ne\\f`g'  (Linux & macOS)
 *      a'b"c$d<ENTER>e\f`g  -->  "a'b`"c`$d`ne\f``g"  (Windows)
 **/
export function shellString(str: string): string {
    return process.platform === 'win32' ? powershellString(str) : bashString(str);
}

/**
 *  Convert a string into quoted and escaped string for bash script. It supports multi-line strings.
 **/
export function bashString(str: string): string {
    // for readability of generated script,
    // use ansi-c quoting when `str` contains single quote or newline,
    // use single quotes otherwise
    if (str.includes(singleQuote) || str.includes(newline)) {
        str = str.replaceAll(backslash, doubleBackslash);
        str = str.replaceAll(singleQuote, backslash + singleQuote);
        str = str.replaceAll(newline, backslash + 'n');
        return '$' + singleQuote + str + singleQuote;
    } else {
        return singleQuote + str + singleQuote;
    }
}

/**
 *  Convert a string into quoted and escaped string for PowerShell script. It supports multi-line strings.
 **/
export function powershellString(str: string): string {
    // for readability and robustness of generated script,
    // use double quotes for multi-line string,
    // use single quotes otherwise
    if (str.includes(newline)) {
        str = str.replaceAll(backtick, doubleBacktick);
        str = str.replaceAll(doubleQuote, backtick + doubleQuote);
        str = str.replaceAll(newline, backtick + 'n');
        str = str.replaceAll('$', backtick + '$');
        return doubleQuote + str + doubleQuote;
    } else {
        str = str.replaceAll(singleQuote, singleQuote + singleQuote);
        return singleQuote + str + singleQuote;
    }
}

export function createScriptFile(path: string, content: string): Promise<void> {
    // eslint-disable-next-line no-control-regex
    if (path.endsWith('.ps1') && !/^[\x00-\x7F]*$/.test(content)) {
        // PowerShell does not use UTF-8 by default.
        // Add BOM to inform it if the script contains non-ASCII characters.
        content = '\uFEFF' + content;
    }
    return fsPromises.writeFile(path, content, { mode: 0o777 });
}
