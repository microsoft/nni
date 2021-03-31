// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

export const CONTAINER_INSTALL_NNI_SHELL_FORMAT: string =
`#!/bin/bash
if python3 -c 'import nni' > /dev/null 2>&1; then
  # nni module is already installed, skip
  :
else
  # Install nni
  python3 -m pip install --user --upgrade nni
fi`;

export const CONTAINER_INSTALL_NNI_SHELL_FORMAT_FOR_WIN: string =
`echo off
python -c "import nni" 2>nul
if not %ERRORLEVEL% EQU 0 (
    python -m pip install --user --upgrade nni
)`;