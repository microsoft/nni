BIN_PATH ?= ${HOME}/.local/bin
INSTALL_PREFIX ?= ${HOME}/.local
EXAMPLES_PATH ?= ${HOME}/nni/examples
WHOAMI := $(shell whoami)
YARN := $(INSTALL_PREFIX)/yarn/bin/yarn
PIP_INSTALL := python3 -m pip install
PIP_UNINSTALL := python3 -m pip uninstall
PIP_MODE ?= --user
ifdef VIRTUAL_ENV
undefine PIP_MODE
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
    ifneq ('$(HOME)', '/root')
        ifeq (${WHOAMI}, root)
			### Sorry, sudo make install is not supported ###
			exit 1
        endif
    endif

	mkdir -p $(BIN_PATH)
	mkdir -p $(INSTALL_PREFIX)/nni
	mkdir -p $(EXAMPLES_PATH)
	
	### Installing NNI Manager ###
	cp -rT src/nni_manager/dist $(INSTALL_PREFIX)/nni/nni_manager
	cp -rT src/nni_manager/node_modules $(INSTALL_PREFIX)/nni/nni_manager/node_modules
	
	### Installing Web UI ###
	cp -rT src/webui/build $(INSTALL_PREFIX)/nni/webui
	ln -sf $(INSTALL_PREFIX)/nni/nni_manager/node_modules/serve/bin/serve.js $(BIN_PATH)/serve
	
	### Installing Python SDK dependencies ###
	$(PIP_INSTALL) $(PIP_MODE) -r src/sdk/pynni/requirements.txt
	### Installing Python SDK ###
	cd src/sdk/pynni && python3 setup.py install $(PIP_MODE)
	
	### Installing nnictl ###
	cd tools && python3 setup.py install $(PIP_MODE)
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd $(INSTALL_PREFIX)/nni/nni_manager && node main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnictl
	echo 'NNI_MANAGER=$(BIN_PATH)/nnimanager WEB_UI_FOLDER=$(INSTALL_PREFIX)/nni/webui python3 -m nnicmd.nnictl $$@' >> $(BIN_PATH)/nnictl
	chmod +x $(BIN_PATH)/nnictl
	
	### Installing examples ###
	cp -rT examples $(EXAMPLES_PATH)


pip-install:
    ifneq ('$(HOME)', '/root')
        ifeq (${WHOAMI}, root)
			### Sorry, sudo pip install is not supported ###
			exit 1
        endif
    endif

	### Prepare Node.js ###
	wget https://nodejs.org/dist/v10.9.0/node-v10.9.0-linux-x64.tar.xz
	tar xf node-v10.9.0-linux-x64.tar.xz
	cp -rT node-v10.9.0-linux-x64 $(INSTALL_PREFIX)/node
	
	### Prepare Yarn 1.9.4 ###
	wget https://github.com/yarnpkg/yarn/releases/download/v1.9.4/yarn-v1.9.4.tar.gz
	tar xf yarn-v1.9.4.tar.gz
	cp -rT yarn-v1.9.4 $(INSTALL_PREFIX)/yarn

	### Export PATH for node and yarn, and build NNI Manager ###
	export PATH=$(INSTALL_PREFIX)/node/bin:$(INSTALL_PREFIX)/yarn/bin:$(PATH) && cd src/nni_manager && $(YARN) && $(YARN) build

	### Building Web UI ###
	export PATH=$(INSTALL_PREFIX)/node/bin:$(INSTALL_PREFIX)/yarn/bin:$(PATH) && cd src/webui && $(YARN) && $(YARN) build
	
	mkdir -p $(BIN_PATH)
	mkdir -p $(INSTALL_PREFIX)/nni
	mkdir -p $(EXAMPLES_PATH)
	
	### Installing NNI Manager ###
	cp -rT src/nni_manager/dist $(INSTALL_PREFIX)/nni/nni_manager
	cp -rT src/nni_manager/node_modules $(INSTALL_PREFIX)/nni/nni_manager/node_modules

	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd $(INSTALL_PREFIX)/nni/nni_manager && node main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager

	echo '#!/bin/sh' > $(BIN_PATH)/nnictl
	echo 'NNI_MANAGER=$(BIN_PATH)/nnimanager WEB_UI_FOLDER=$(INSTALL_PREFIX)/nni/webui python3 -m nnicmd.nnictl $$@' >> $(BIN_PATH)/nnictl
	chmod +x $(BIN_PATH)/nnictl

	### Installing Web UI ###
	cp -rT src/webui/build $(INSTALL_PREFIX)/nni/webui
	ln -sf $(INSTALL_PREFIX)/nni/nni_manager/node_modules/serve/bin/serve.js $(BIN_PATH)/serve
	
	### Installing examples ###
	cp -rT examples $(EXAMPLES_PATH)


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
	$(PIP_INSTALL) $(PIP_MODE) -r src/sdk/pynni/requirements.txt
	### Installing Python SDK ###
	cd src/sdk/pynni && $(PIP_INSTALL) $(PIP_MODE) -e .
	
	### Installing nnictl ###
	cd tools && $(PIP_INSTALL) $(PIP_MODE) -e .
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd $(INSTALL_PREFIX)/nni/nni_manager && node main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnictl
	echo 'NNI_MANAGER=$(BIN_PATH)/nnimanager python3 -m nnicmd.nnictl $$@' >> $(BIN_PATH)/nnictl
	chmod +x $(BIN_PATH)/nnictl
	
	### Installing examples ###
	ln -sf $(EXAMPLES_PATH) $(PWD)/examples


uninstall:
	-$(PIP_UNINSTALL) -y nni
	-$(PIP_UNINSTALL) -y nnictl
	-rm -r $(INSTALL_PREFIX)/nni
	-rm -r $(EXAMPLES_PATH)
	-rm $(BIN_PATH)/serve
	-rm $(BIN_PATH)/nnimanager
	-rm $(BIN_PATH)/nnictl
