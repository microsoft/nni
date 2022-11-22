'use strict';

process.env.BABEL_ENV = 'production';
process.env.NODE_ENV = 'production';

process.on('unhandledRejection', err => { throw err; });

require('./config/env');


const fsPromises = require('fs/promises');
const fsExtra = require('fs-extra');
const FileSizeReporter = require('react-dev-utils/FileSizeReporter');
const { checkBrowsers } = require('react-dev-utils/browsersHelper');
const chalk = require('react-dev-utils/chalk');
const printBuildError = require('react-dev-utils/printBuildError');
const webpack = require('webpack');

const formatWebpackMessages = require('./formatWebpackMessages');
const configFactory = require('./config/webpack.config');
const paths = require('./config/paths');


async function main() {
    await checkBrowsers(paths.appPath, process.stdout.isTTY);

    const previousFileSizes = await FileSizeReporter.measureFileSizesBeforeBuild(paths.appBuild);

    await fsPromises.rm(paths.appBuild, { force: true, recursive: true });
    await fsExtra.copy(paths.appPublic, paths.appBuild, {  // fs.cp() seems not work
        dereference: true,
        filter: file => (file !== paths.appHtml)
    });

    console.log('Creating an optimized production build...');

    const config = configFactory('production');
    const compiler = webpack(config);
    const result = await asyncRun(compiler);

    const { errors, warnings } = formatWebpackMessages(result);

    if (errors.length) {
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

    console.log('File sizes after gzip:\n');
    FileSizeReporter.printFileSizesAfterBuild(
        result.stats,
        previousFileSizes,
        paths.appBuild,
        512 * 1024,  // WARN_AFTER_BUNDLE_GZIP_SIZE
        1024 * 1024  // WARN_AFTER_CHUNK_GZIP_SIZE
    );
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
