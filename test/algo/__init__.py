# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extra tests for "algorithms", complementary to ut.

If the test satisfies one of the following conditions, it should be put here:

1. The test could take a while to finish.
2. The test doesn't work on the free agent. It needs accelerators like GPUs.
3. The test is dedicated for a specific replacable module, which doesn't involve core functionalities.

Note that if a core function (something, without which NNI doesn't work at all) were to be tested against,
it's still highly recommended to write at least a simple test in UT.
If comprehensive tests are expensive, they can belong here.
"""

# Import ut to set environment variables
import ut
