// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert';
import { getIPV4Address } from '../../common/utils';

it('getIpv4Address', async () => {
    const ip1 = await getIPV4Address();
    const ip2 = await getIPV4Address();
    assert.match(ip1, /^\d+\.\d+\.\d+\.\d+$/);
    assert.equal(ip1, ip2);
});
