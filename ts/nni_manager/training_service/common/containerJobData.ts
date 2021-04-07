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
`python -c "import nni" 2>$error
if ($error -ne ''){
  python -m pip install --user --upgrade nni
}
exit`;