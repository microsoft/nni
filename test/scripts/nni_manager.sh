#!/bin/bash

# -------------For typescrip unittest-------------
cd ../../src/nni_manager
echo "Testing: nni_manager..."
npm run test
cd ${CWD}