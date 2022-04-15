'use strict';

process.env.BABEL_ENV = 'development';
process.env.NODE_ENV = 'development';

process.on('unhandledRejection', err => { throw err; });

require('./config/env');


const fsPromises = require('fs/promises');
const fsExtra = require('fs-extra');
const chalk = require('react-dev-utils/chalk');
const printBuildError = require('react-dev-utils/printBuildError');
const webpack = require('webpack');

const configFactory = require('./config/webpack.config.dev');
const paths = require('./config/paths');
const formatWebpackMessages = require('./formatWebpackMessages');


async function main() {
    await fsPromises.rm(paths.appBuild, { force: true, recursive: true });
    await fsExtra.copy(paths.appPublic, paths.appBuild, {  // fs.cp() seems not work
        dereference: true,
        filter: file => (file !== paths.appHtml)
    });

    console.log('Creating an development build...');

    const config = configFactory('development');
    const compiler = webpack(config);
    const result = await asyncRun(compiler);
    const { errors, warnings } = formatWebpackMessages(result);

    if (errors.length) {
        console.log(JSON.stringify(errors));
        console.log(chalk.red('Failed to compile.\n'));
        printBuildError(new Error(errors[0]));
        process.exit(1);
    }

    if (warnings.length) {
        console.log(chalk.yellow('Compiled with warnings.\n'));
        console.log(warnings.join('\n\n'));

        if (isCi()) {
            process.exit(1);
        }

    } else {
        console.log(chalk.green('Compiled successfully.\n'));
    }
}

function asyncRun(compiler) {
    return new Promise((resolve, reject) => {
        compiler.run((err, stats) => {
            resolve({ err, stats });
        });
    });
}

function isCi() {
    if (!process.env.CI) {
        return false;
    }
    if (typeof process.env.CI !== 'string') {
        return true;
    }
    return process.env.CI.toLowerCase() !== 'false';
}

main().catch(err => {
    console.log(chalk.red('Failed to compile.\n'));
    printBuildError(err);
    process.exit(1);
});
