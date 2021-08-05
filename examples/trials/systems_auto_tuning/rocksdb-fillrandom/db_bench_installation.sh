#!/bin/bash
# Install db_bench and its dependencies on Ubuntu

pushd $PWD 1>/dev/null

# install snappy
echo "****************** Installing snappy *******************"
sudo apt-get install libsnappy-dev -y

# install gflag
echo "****************** Installing gflag ********************"
cd /tmp
git clone https://github.com/gflags/gflags.git
cd gflags
git checkout v2.0
./configure && make && sudo make install

# install rocksdb
echo "****************** Installing rocksdb ******************"
cd /tmp
git clone https://github.com/facebook/rocksdb.git
cd rocksdb
CPATH=/usr/local/include LIBRARY_PATH=/usr/local/lib DEBUG_LEVEL=0 make db_bench -j7

DIR=$HOME/.local/bin/
if [[ ! -e $DIR ]]; then
    mkdir $dir
elif [[ ! -d $DIR ]]; then
    echo "$DIR already exists but is not a directory" 1>&2
    exit
fi
mv db_bench $HOME/.local/bin &&
echo "Successfully installed rocksed in "$DIR" !" &&
echo "Please add "$DIR" to your PATH for runing this example."

popd 1>/dev/null
