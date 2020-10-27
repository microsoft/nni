import * as glob from 'glob';

// Istanbul only generates report for used/imported files, the files are not used/imported by test cases
// are not included in code coverage reports.
// This is a workaround to import all files in order to show all source files in code coverage reports.

glob.sync('**/*.ts').forEach((file) => {
    if (file.indexOf('node_modules/') < 0 && file.indexOf('types/') < 0
        && file.indexOf('.test.ts') < 0 && file.indexOf('main.ts')) {
        try {
            import('../../' + file);
        } catch(err) {
        }
    }
})
