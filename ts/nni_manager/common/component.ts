// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import * as ioc from 'typescript-ioc';

const Inject: (...args: any[]) => any = ioc.Inject;
const Singleton: (target: any) => void = ioc.Singleton;
const Container = ioc.Container;
const Provides = ioc.Provides;

function get<T>(source: any): T {
    return ioc.Container.get(source) as T;
}

export { Provides, Container, Inject, Singleton, get };
