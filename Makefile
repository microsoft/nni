ifeq (`id -u`, 0)  # is root 
    BIN_PATH ?= /usr/bin
    INSTALL_PREFIX ?= /usr/share
else  # is normal user
    BIN_PATH ?= ${HOME}/.local/bin
    INSTALL_PREFIX ?= ${HOME}/.local
    PIP_MODE ?= --user
endif


.PHONY: build install uninstall dev-install

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
	mkdir -p $(BIN_PATH)
	mkdir -p $(INSTALL_PREFIX)/nni
	
	### Installing NNI Manager ###
	cp -rT src/nni_manager/dist $(INSTALL_PREFIX)/nni/nni_manager
	cp -rT src/nni_manager/node_modules $(INSTALL_PREFIX)/nni/nni_manager/node_modules
	
	### Installing Web UI ###
	cp -rT src/webui/build $(INSTALL_PREFIX)/nni/webui
	ln -sf $(INSTALL_PREFIX)/nni/nni_manager/node_modules/serve/bin/serve.js $(BIN_PATH)/serve
	
	### Installing Python SDK dependencies ###
	pip3 install $(PIP_MODE) -r src/sdk/pynni/requirements.txt
	### Installing Python SDK ###
	cd src/sdk/pynni && python3 setup.py install $(PIP_MODE)
	
	### Installing nnictl ###
	cd tools && python3 setup.py install $(PIP_MODE)
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd $(INSTALL_PREFIX)/nni/nni_manager && node main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnictl
	echo 'NNI_MANAGER=$(BIN_PATH)/nnimanager python3 -m nnicmd.nnictl $$@' >> $(BIN_PATH)/nnictl
	chmod +x $(BIN_PATH)/nnictl
	
	### Installing examples ###
	cp -rT examples $(INSTALL_PREFIX)/nni/examples


pip-install:
	### Prepare Node.js ###
	wget https://nodejs.org/dist/v10.9.0/node-v10.9.0-linux-x64.tar.xz
	tar xf node-v10.9.0-linux-x64.tar.xz
	sudo mkdir -p /usr/local/node/
	sudo rm -rf /usr/local/node/*
	sudo cp node-v10.9.0-linux-x64/* /usr/local/node/
	
	### Prepare Yarn 1.9.4 ###
	wget https://github.com/yarnpkg/yarn/releases/download/v1.9.4/yarn-v1.9.4.tar.gz
	tar xf yarn-v1.9.4.tar.gz
	sudo mkdir -p /usr/local/yarn/
	sudo rm -rf /usr/local/yarn/*
	sudo cp yarn-v1.9.4/* /usr/local/yarn/

	### Building NNI Manager ###
	cd src/nni_manager && /usr/local/yarn/bin/yarn && /usr/local/yarn/bin/yarn build
	
	### Building Web UI ###
	cd src/webui && /usr/local/yarn/bin/yarn && /usr/local/yarn/bin/yarn build
	
	mkdir -p $(NODE_PATH)/nni
	mkdir -p $(EXAMPLE_PATH)
	
	### Installing NNI Manager ###
	cp -rT src/nni_manager/dist $(NODE_PATH)/nni/nni_manager
	cp -rT src/nni_manager/node_modules $(NODE_PATH)/nni/nni_manager/node_modules
	
	### Installing Web UI ###
	cp -rT src/webui/build $(NODE_PATH)/nni/webui
	ln -sf $(NODE_PATH)/nni/nni_manager/node_modules/serve/bin/serve.js $(BIN_PATH)/serve

	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd $(NODE_PATH)/nni/nni_manager && node main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager
	
	### Installing examples ###
	cp -rT examples $(EXAMPLE_PATH)


dev-install:
	mkdir -p $(BIN_PATH)
	mkdir -p $(INSTALL_PREFIX)/nni
	
	### Installing NNI Manager ###
	ln -sf $(INSTALL_PREFIX)/nni/nni_manager $(PWD)/src/nni_manager/dist
	ln -sf $(INSTALL_PREFIX)/nni/nni_manager/node_modules $(PWD)/src/nni_manager/node_modules
	
	### Installing Web UI ###
	ln -sf $(INSTALL_PREFIX)/nni/webui $(PWD)/src/webui
	ln -sf $(INSTALL_PREFIX)/nni/nni_manager/node_modules/serve/bin/serve.js $(BIN_PATH)/serve
	
	### Installing Python SDK dependencies ###
	pip3 install $(PIP_MODE) -r src/sdk/pynni/requirements.txt
	### Installing Python SDK ###
	cd src/sdk/pynni && pip3 install $(PIP_MODE) -e .
	
	### Installing nnictl ###
	cd tools && pip3 install $(PIP_MODE) -e .
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd $(INSTALL_PREFIX)/nni/nni_manager && node main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnictl
	echo 'NNI_MANAGER=$(BIN_PATH)/nnimanager python3 -m nnicmd.nnictl $$@' >> $(BIN_PATH)/nnictl
	chmod +x $(BIN_PATH)/nnictl
	
	### Installing examples ###
	ln -sf $(INSTALL_PREFIX)/nni/examples $(PWD)/examples


uninstall:
	-pip3 uninstall -y nni
	-pip3 uninstall -y nnictl
	-rm -r $(INSTALL_PREFIX)/nni
	-rm $(BIN_PATH)/serve
	-rm $(BIN_PATH)/nnimanager
	-rm $(BIN_PATH)/nnictl
