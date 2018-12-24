#!/bin/bash

# -------------For typescrip unittest-------------
cd ../../src/nni_manager
echo ""
echo "===========================Testing: nni_manager==========================="
sed -ie 's/NNI_VERSION/1.0.0/' package.json
npm run test