// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

export function generateChannelName(url: string): string {
    const parts = url.split('/');
    for (let i = parts.length - 1; i > 1; i--) {
        if (parts[i]) {
            return parts[i];
        }
    }
    return 'anonymous';
}
