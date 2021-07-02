// for readability
const singleQuote = "'";
const doubleQuote = '"';
const backtick = '`';
const backslash = '\\';
const doubleBacktick = '``';
const doubleBackslash = '\\\\';
const newline = '\n';

export function shellString(str: string): string {
    return process.platform === 'win32' ? pwshString(str) : bashString(str);
}

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

export function pwshString(str: string): string {
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
