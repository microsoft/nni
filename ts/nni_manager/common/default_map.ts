// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

export class DefaultMap<K, V> extends Map<K, V> {
    private defaultFactory: () => V;

    constructor(defaultFactory: () => V) {
        super();
        this.defaultFactory = defaultFactory;
    }

    public get(key: K): V {
        const value = super.get(key);
        if (value !== undefined) {
            return value;
        }

        const defaultValue = this.defaultFactory();
        this.set(key, defaultValue);
        return defaultValue;
    }
}
