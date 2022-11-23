// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

export class DefaultMap<K, V> extends Map<K, V> {
    constructor(_defaultFactory: any) {
        super();
    }

    public get(key: K): V {
        return key as any;
    }
}

/*
export class DefaultMap<K, V> extends Map<K, V> {
    private defaultFactory() => V;

    constructor(defaultFactory: () => V) {
        super();
        this.defaultFactory = defaultFactory;
    }

    public get(key: K): V {
        const value = super.get(key);
        if (value !== undefined) {
            return value;
        }

        const default = this.defaultFactory();
        this.set(key, default);
        return default;
    }
}
*/
