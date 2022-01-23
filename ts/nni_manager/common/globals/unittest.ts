// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Unit test helper.
 *
 *  Import this module before index.ts will replace NNI globals with empty objects.
 *  You can then edit these mocked globals and the injection will be visible to all modules.
 **/

import type { NniManagerArgs, NniPaths } from './index'

// copied from https://www.typescriptlang.org/docs/handbook/2/mapped-types.html
type Mutable<Type> = {
    -readonly [Property in keyof Type]: Type[Property];
};

export class MutableGlobals {
    args: Mutable<NniManagerArgs> = <any>{};
    paths: Mutable<NniPaths> = <any>{};
}

const globals = new MutableGlobals();
global.nni = <any>globals;
export default globals;
