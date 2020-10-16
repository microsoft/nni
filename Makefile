SHELL := /bin/bash

_INFO := $(shell echo -e '\033[1;36m')
_END := $(shell echo -e '\033[0m')

## Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
    OS_SPEC := linux
else ifeq ($(UNAME_S), Darwin)
    OS_SPEC := darwin
else
    $(error platform $(UNAME_S) not supported)
endif

_PWD := $(PWD)
YARN ?= PATH=$(_PWD)/nni_node:$${PATH} $(PWD)/toolchain/yarn/bin/yarn


.PHONY: build
build: nni_node/node create-symlinks
	#$(_INFO) Building NNI Manager $(_END)
	cd ts/nni_manager && $(YARN) && $(YARN) build
	cp -rf ts/nni_manager/config ts/nni_manager/dist/
	#$(_INFO) Building WebUI $(_END)
	cd ts/webui && $(YARN) && $(YARN) build
	#$(_INFO) Building NAS UI $(_END)
	cd ts/nasui && $(YARN) && $(YARN) build


nni_node/node:
	mkdir -p toolchain
	wget https://nodejs.org/dist/v10.22.1/node-v10.22.1-$(OS_SPEC)-x64.tar.xz -O toolchain/node.tar.xz
	wget https://github.com/yarnpkg/yarn/releases/download/v1.22.10/yarn-v1.22.10.tar.gz -O toolchain/yarn.tar.gz
	
	mkdir -p toolchain/node toolchain/yarn
	tar -xf toolchain/node.tar.xz -C toolchain/node --strip-components 1
	tar -xf toolchain/yarn.tar.gz -C toolchain/yarn --strip-components 1
	
	cp toolchain/node/bin/node nni_node/


.PHONY: create-symlinks
create-symlinks:
	ln -sfT ../ts/nni_manager/dist/common nni_node/common
	ln -sfT ../ts/nni_manager/dist/config nni_node/config
	ln -sfT ../ts/nni_manager/dist/core nni_node/core
	ln -sfT ../ts/nni_manager/dist/rest_server nni_node/rest_server
	ln -sfT ../ts/nni_manager/dist/training_service nni_node/training_service
	ln -sfT ../ts/nni_manager/dist/main.js nni_node/main.js
	ln -sfT ../ts/nni_manager/package.json nni_node/package.json
	ln -sfT ../ts/nni_manager/node_modules nni_node/node_modules
	
	ln -sfT ../ts/webui/build nni_node/static
	
	mkdir -p nni_node/nasui
	ln -sfT ../../ts/nasui/build nni_node/nasui/build
	ln -sfT ../../ts/nasui/server.js nni_node/nasui/server.js


.PHONY: clean
clean:
	-rm nni_node/node
	-rm -rf toolchain
	-rm -rf nni_node/common
	-rm -rf nni_node/config
	-rm -rf nni_node/core
	-rm -rf nni_node/rest_server
	-rm -rf nni_node/training_service
	-rm -rf nni_node/main.js
	-rm -rf nni_node/package.json
	-rm -rf nni_node/node_modules
	-rm -rf nni_node/nasui
	-rm -rf nni_node/static
