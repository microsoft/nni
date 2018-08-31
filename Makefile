BIN_PATH ?= /usr/bin
NODE_PATH ?= /usr/share
EXAMPLE_PATH ?= /usr/share/nni/examples

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
	mkdir -p $(NODE_PATH)/nni
	mkdir -p $(EXAMPLE_PATH)
	
	### Installing NNI Manager ###
	cp -rT src/nni_manager/dist $(NODE_PATH)/nni/nni_manager
	cp -rT src/nni_manager/node_modules $(NODE_PATH)/nni/nni_manager/node_modules
	
	### Installing Web UI ###
	cp -rT src/webui/build $(NODE_PATH)/nni/webui
	ln -sf $(NODE_PATH)/nni/nni_manager/node_modules/serve/bin/serve.js $(BIN_PATH)/serve
	
	### Installing Python SDK dependencies ###
	pip3 install -r src/sdk/pynni/requirements.txt
	### Installing Python SDK ###
	cd src/sdk/pynni && python3 setup.py install
	
	### Installing nnictl ###
	cd tools && python3 setup.py install
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd $(NODE_PATH)/nni/nni_manager && node main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager
	
	install -m 755 tools/nnictl $(BIN_PATH)/nnictl
	
	### Installing examples ###
	cp -rT examples $(EXAMPLE_PATH)


dev-install:
	### Installing Python SDK dependencies ###
	pip3 install --user -r src/sdk/pynni/requirements.txt
	### Installing Python SDK ###
	cd src/sdk/pynni && pip3 install --user -e .
	
	### Installing nnictl ###
	cd tools && pip3 install --user -e .


uninstall:
	-rm -r $(EXAMPLE_PATH)
	-rm -r $(NODE_PATH)/nni
	-pip3 uninstall -y nnictl
	-pip3 uninstall -y nni
	-rm $(BIN_PATH)/nnictl
	-rm $(BIN_PATH)/nnimanager
	-rm $(BIN_PATH)/serve
