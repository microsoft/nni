DST := ${PWD}/debian/nni
SRC := ../../..

BIN_PATH := $(DST)/usr/bin
INSTALL_PREFIX := $(DST)/usr/share
BASH_COMP_SCRIPT := $(DST)/usr/share/bash-completion/completions/nnictl

NNI_MAKE := BIN_PATH=$(BIN_PATH) INSTALL_PREFIX=$(INSTALL_PREFIX) BASH_COMP_SCRIPT=$(BASH_COMP_SCRIPT) $(MAKE) -C $(SRC)

build:
	$(NNI_MAKE) build

clean:
	# TODO

install:
	$(NNI_MAKE) install-dependencies
	$(NNI_MAKE) install-install
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd /usr/share/nni/nni_manager' >> $(BIN_PATH)/nnimanager
	echo '/usr/share/nni/node/bin/node main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnictl
	echo 'NNI_SERVE=/usr/share/nni/serve/serve \' >> $(BIN_PATH)/nnictl
	echo 'WEB_UI_FOLDER=/usr/share/nni/webui \' >> $(BIN_PATH)/nnictl
	echo 'python3 -m nnicmd.nnictl $$@' >> $(BIN_PATH)/nnictl
	chmod +x $(BIN_PATH)/nnictl
