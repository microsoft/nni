// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

export const CONTAINER_INSTALL_NNI_SHELL_FORMAT: string =
`#!/bin/bash
if python3 -c 'import nni' > /dev/null 2>&1; then
  # nni module is already installed, skip
  return
else
  # Install nni
  python3 -m pip install --user --upgrade nni
fi`;

export const AML_CONTAINER_INSTALL_NNI_SHELL_FORMAT: string =
`#!/bin/bash
if python3 -c 'import nni' > /dev/null 2>&1; then
  # nni module is already installed, skip
  return
else
  # Install nni
  python3 -m pip install --user --no-cache-dir -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nni==1.680
fi`;


