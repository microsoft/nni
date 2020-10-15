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
	cd nni_node/nni_manager && $(YARN) && $(YARN) build
	cp -rf nni_node/nni_manager/config nni_node/nni_manager/dist/
	#$(_INFO) Building WebUI $(_END)
	cd nni_node/webui && $(YARN) && $(YARN) build
	#$(_INFO) Building NAS UI $(_END)
	cd nni_node/nasui && $(YARN) && $(YARN) build


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
	mkdir -p nni_node/build
	ln -sfT ../nni_manager/dist/common nni_node/build/common
	ln -sfT ../nni_manager/dist/config nni_node/build/config
	ln -sfT ../nni_manager/dist/core nni_node/build/core
	ln -sfT ../nni_manager/dist/rest_server nni_node/build/rest_server
	ln -sfT ../nni_manager/dist/training_service nni_node/build/training_service
	ln -sfT ../nni_manager/dist/main.js nni_node/build/main.js
	ln -sfT ../nni_manager/dist/package.json nni_node/build/package.json
	ln -sfT ../nni_manager/node_modules nni_node/build/node_modules
	
	ln -sfT ../webui/build nni_node/build/static
	
	mkdir -p nni_node/build/nasui
	ln -sfT ../nasui/build nni_node/build/nasui/build
	ln -sfT ../nasui/server.js nni_node/build/nasui/server.js


.PHONY: clean
clean:
	-rm nni_node/node
	-rm -rf toolchain
	-rm -rf nni_node/build
	-rm -rf nni_node/nni_manager/dist
	-rm -rf nni_node/nni_manager/node_modules
	-rm -rf nni_node/webui/build
	-rm -rf nni_node/webui/node_modules
	-rm -rf nni_node/nasui/build
	-rm -rf nni_node/nasui/node_modules
