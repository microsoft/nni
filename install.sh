#!/bin/bash
wget -4 -nc https://nodejs.org/dist/v10.9.0/node-v10.9.0-linux-x64.tar.xz --header "Referer: nodejs.org"
tar -xf 'node-v10.9.0-linux-x64.tar.xz'
sudo cp -rf node-v10.9.0-linux-x64/* /usr/local/node/
rm -rf  node-v10.9.0-linux-x64*
wget -4 -nc https://github.com/yarnpkg/yarn/releases/download/v1.9.4/yarn-v1.9.4.tar.gz
tar -xf 'yarn-v1.9.4.tar.gz'
sudo cp -rf yarn-v1.9.4/* /usr/local/yarn/
rm -rf yarn-v1.9.4*
export PATH=/usr/local/node/bin:/usr/local/yarn/bin:$PATH
make
sudo make install