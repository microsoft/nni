#!/bin/bash
make install-dependencies
make build
make dev-install
make install-examples
make update-bash-config
source ~/.bashrc
