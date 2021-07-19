// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import * as assert from 'assert';
import { getIPV4Address } from '../../common/utils';

it('getIpv4Address', async () => {
    const ip = await getIPV4Address();
    assert.match(ip, /^\d+\.\d+\.\d+\.\d+$/)
});
