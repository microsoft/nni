Installation instructions
===

## install using deb file

TBD

## install from source code
   * Prepare Node.js 10.9.0 or above
   
         wget https://nodejs.org/dist/v10.9.0/node-v10.9.0-linux-x64.tar.xz
         tar xf node-v10.9.0-linux-x64.tar.xz
         mv node-v10.9.0-linux-x64/* /usr/local/node/
   * Prepare Yarn 1.9.4 or above

         wget https://github.com/yarnpkg/yarn/releases/download/v1.9.4/yarn-v1.9.4.tar.gz
         tar xf yarn-v1.9.4.tar.gz
         mv yarn-v1.9.4/* /usr/local/yarn/
   * Add Node.js and Yarn in PATH

         export PATH=/usr/local/node/bin:/usr/local/yarn/bin:$PATH
   * clone nni source code

         git clone https://github.com/Microsoft/NeuralNetworkIntelligence
   * build and install nni

         make build
         sudo make install
