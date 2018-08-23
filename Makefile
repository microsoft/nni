BIN_PATH ?= $(HOME)/.nni/bin/
NNI_PATH ?= $(HOME)/.nni/

SRC_DIR := ${PWD}

.PHONY: build install uninstall

build:
	### Building NNI Manager ###
	cd src/nni_manager && yarn && yarn build
	
	### Building Web UI ###
	cd src/webui && yarn && yarn build
	
	### Building Python SDK ###
	cd src/sdk/pynni && python3 setup.py build
	
	### Building nnictl ###
	cd tools && python3 setup.py build


install:
	mkdir -p $(NNI_PATH)
	mkdir -p $(BIN_PATH)
	
	### Installing NNI Manager ###
	cp -rT src/nni_manager/dist $(NNI_PATH)nni_manager
	cp -rT src/nni_manager/node_modules $(NNI_PATH)nni_manager/node_modules
	
	### Installing Web UI ###
	cp -rT src/webui/build $(NNI_PATH)webui
	ln -sf $(NNI_PATH)nni_manager/node_modules/serve/bin/serve.js $(BIN_PATH)serve
	
	### Installing Python SDK dependencies ###
	pip3 install -r src/sdk/pynni/requirements.txt
	### Installing Python SDK ###
	cd src/sdk/pynni && pip3 install -e .
	
	### Installing nnictl ###
	cd tools && pip3 install -e .
	
	echo '#!/bin/sh' > $(BIN_PATH)nnimanager
	echo 'cd $(NNI_PATH)nni_manager && node main.js $$@' >> $(BIN_PATH)nnimanager
	chmod +x $(BIN_PATH)nnimanager
	
	install -m 755 tools/nnictl $(BIN_PATH)nnictl
	
	### Installing examples ###
	cp -rT examples $(NNI_PATH)examples


dev-install:
	### Installing Python SDK dependencies ###
	pip3 install -r src/sdk/pynni/requirements.txt
	### Installing Python SDK ###
	cd src/sdk/pynni && pip3 install -e .
	
	### Installing nnictl ###
	cd tools && pip3 install -e .


uninstall:
	-rm -r $(NNI_PATH)
	-rm -r $(BIN_PATH)
	-pip3 uninstall -y nnictl
	-pip3 uninstall -y nni

