#!/bin/bash
INSTALL_PREFIX=${HOME}/.local
wget -4 -nc https://nodejs.org/dist/v10.9.0/node-v10.9.0-linux-x64.tar.xz --header "Referer: nodejs.org"
tar -xf 'node-v10.9.0-linux-x64.tar.xz'
sudo cp -rT node-v10.9.0-linux-x64 ${INSTALL_PREFIX}/node
rm -rf  node-v10.9.0-linux-x64*
wget -4 -nc https://github.com/yarnpkg/yarn/releases/download/v1.9.4/yarn-v1.9.4.tar.gz
tar -xf 'yarn-v1.9.4.tar.gz'
sudo cp -rT yarn-v1.9.4 ${INSTALL_PREFIX}/yarn
rm -rf yarn-v1.9.4*
export PATH=${INSTALL_PREFIX}/node/bin:${INSTALL_PREFIX}/yarn/bin:$PATH
make
make install
