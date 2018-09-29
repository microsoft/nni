#!/bin/bash
make install-dependencies
make build
make dev-install
make install-examples
source ~/.bashrc
