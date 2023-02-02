# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
