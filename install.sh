#!/bin/bash
INSTALL_PREFIX=${HOME}/.local
mkdir -p ${INSTALL_PREFIX}
wget -4 -nc https://nodejs.org/dist/v10.9.0/node-v10.9.0-linux-x64.tar.xz --header "Referer: nodejs.org"
tar -xf 'node-v10.9.0-linux-x64.tar.xz'
cp -rT node-v10.9.0-linux-x64 ${INSTALL_PREFIX}/node
rm -rf  node-v10.9.0-linux-x64*
wget -4 -nc https://github.com/yarnpkg/yarn/releases/download/v1.9.4/yarn-v1.9.4.tar.gz
tar -xf 'yarn-v1.9.4.tar.gz'
cp -rT yarn-v1.9.4 ${INSTALL_PREFIX}/yarn
rm -rf yarn-v1.9.4*
NODE_BIN=${INSTALL_PREFIX}/node/bin
YARN_BIN=${INSTALL_PREFIX}/yarn/bin
export PATH=${INSTALL_PREFIX}/node/bin:${INSTALL_PREFIX}/yarn/bin:$PATH
echo $PATH|grep -q ${NODE_BIN} || echo "export PATH=${NODE_BIN}:\${PATH}" >> ${HOME}/.bashrc
echo $PATH|grep -q ${YARN_BIN} || echo "export PATH=${YARN_BIN}:\${PATH}" >> ${HOME}/.bashrc
source ${HOME}/.bashrc
make
make install