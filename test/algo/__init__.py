# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Extra tests for "algorithms", complementary to UT.

If the test satisfies one of the following conditions, it should be put here:

1. The test could take a while to finish.
2. The test doesn't work on the free agent. It needs accelerators like GPUs.
3. The test is dedicated for a specific replacable module, which doesn't involve core functionalities.

Note that if a test is to ensure the correctness of a "core function", without which NNI doesn't work at all,
it's still highly recommended to include at least a simple test in UT.
If a set of exhaustive tests were to be expensive, they can still belong here.
"""

# Import ut to set environment variables
import ut
