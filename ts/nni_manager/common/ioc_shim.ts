// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'node:assert/strict';

type AbstractClass = {
    name: string;
};

type Class = {
    name: string;
    new(): any;
};

class IocShimClass {
    private singletons: Map<string, any> = new Map();
    private snapshots: Map<string, any> = new Map();

    public bind(keyClass: AbstractClass, valueClass: Class): void {
        const key = keyClass.name;
        assert.ok(!this.singletons.has(key));
        this.singletons.set(key, new valueClass());
    }

    public bindInstance(keyClass: AbstractClass, value: any): void {
        const key = keyClass.name;
        assert.ok(!this.singletons.has(key));
        this.singletons.set(key, value);
    }

    public get<T>(keyClass: AbstractClass): T {
        const key = keyClass.name;
        assert.ok(this.singletons.has(key));
        return this.singletons.get(key);
    }

    public snapshot(keyClass: AbstractClass): void {
        const key = keyClass.name;
        const value = this.singletons.get(key);
        this.snapshots.set(key, value);
    }

    public restore(keyClass: AbstractClass): void {
        const key = keyClass.name;
        const value = this.snapshots.get(key);
        this.singletons.set(key, value);
    }

    // NOTE: for unit test only
    public clear(): void {
        this.singletons.clear();
    }
}

export const IocShim: IocShimClass = new IocShimClass();
