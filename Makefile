# Setting variables

PIP_INSTALL := python3 -m pip install
PIP_UNINSTALL := python3 -m pip uninstall

# detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
	OS_SPEC := linux
	## Colorful output
	_INFO := $(shell echo -e '\e[1;36m')
	_WARNING := $(shell echo -e '\e[1;33m')
	_END := $(shell echo -e '\e[0m')
else ifeq ($(UNAME_S), Darwin)
	OS_SPEC := darwin
else
	$(error platform $(UNAME_S) not supported)
endif



## Install directories
ifeq ($(shell id -u), 0)  # is root
    _ROOT := 1
    ROOT_FOLDER ?= $(shell python3 -c 'import site; from pathlib import Path; print(Path(site.getsitepackages()[0]).parents[2])')
    BASH_COMP_PREFIX ?= /usr/share/bash-completion/completions
else  # is normal user
    ROOT_FOLDER ?= $(shell python3 -c 'import site; from pathlib import Path; print(Path(site.getusersitepackages()).parents[2])')
    ifndef VIRTUAL_ENV
        PIP_MODE ?= --user
    endif
    BASH_COMP_PREFIX ?= ${HOME}/.bash_completion.d
endif
BASH_COMP_SCRIPT := $(BASH_COMP_PREFIX)/nnictl

NNI_INSTALL_PATH ?= $(INSTALL_PREFIX)/nni
NNI_TMP_PATH ?= /tmp

BIN_FOLDER ?= $(ROOT_FOLDER)/bin
NNI_PKG_FOLDER ?= $(ROOT_FOLDER)/nni

## Dependency information
NNI_NODE_TARBALL ?= /tmp/nni-node-$(OS_SPEC)-x64.tar.xz
NNI_NODE_FOLDER = /tmp/nni-node-$(OS_SPEC)-x64
NNI_NODE ?= $(BIN_FOLDER)/node
NNI_YARN_TARBALL ?= /tmp/nni-yarn.tar.gz
NNI_YARN_FOLDER ?= /tmp/nni-yarn
NNI_YARN := PATH=$(BIN_FOLDER):$${PATH} $(NNI_YARN_FOLDER)/bin/yarn

# Main targets

.PHONY: build
build:
	#$(_INFO) Building NNI Manager $(_END)
	cd src/nni_manager && $(NNI_YARN) && $(NNI_YARN) build
	#$(_INFO) Building WebUI $(_END)
	cd src/webui && $(NNI_YARN) && $(NNI_YARN) build
	#$(_INFO) Building Python SDK $(_END)
	cd src/sdk/pynni && python3 setup.py build
	#$(_INFO) Building nnictl $(_END)
	cd tools && python3 setup.py build

# All-in-one target for non-expert users
# Installs NNI as well as its dependencies, and update bashrc to set PATH
.PHONY: easy-install
easy-install: check-perm
easy-install: install-dependencies
easy-install: build
easy-install: install
easy-install: update-bash-config
easy-install:
	#$(_INFO) Complete! $(_END)

# All-in-one target for developer users
# Install NNI as well as its dependencies, and update bashrc to set PATH
.PHONY: dev-easy-install
dev-easy-install: dev-check-perm
dev-easy-install: install-dependencies
dev-easy-install: build
dev-easy-install: dev-install
dev-easy-install: update-bash-config
dev-easy-install:
	#$(_INFO) Complete! $(_END)

# Standard installation target
# Must be invoked after building
.PHONY: install
install: install-python-modules
install: install-node-modules
install: install-scripts
install:
	#$(_INFO) Complete! You may want to add $(BIN_FOLDER) to your PATH environment $(_END)

# Target for NNI developers
# Creates symlinks instead of copying files
.PHONY: dev-install
dev-install: dev-install-python-modules
dev-install: dev-install-node-modules
dev-install: install-scripts
dev-install:
	#$(_INFO) Complete! You may want to add $(BIN_FOLDER) to your PATH environment $(_END)

# Target for setup.py
# Do not invoke this manually
.PHONY: pip-install
pip-install: install-dependencies
pip-install: build
pip-install: install-node-modules
pip-install: install-scripts
pip-install: update-bash-config

.PHONY: uninstall
uninstall:
	-$(PIP_UNINSTALL) -y nni
	-$(PIP_UNINSTALL) -y nnictl
	-rm -rf $(NNI_PKG_FOLDER)
	-rm -f $(BIN_FOLDER)/node
	-rm -f $(BIN_FOLDER)/nnictl
	-rm -f $(BASH_COMP_SCRIPT)

.PHONY: clean
clean:
	-rm -rf tools/build
	-rm -rf tools/nnictl.egg-info
	-rm -rf src/nni_manager/dist
	-rm -rf src/nni_manager/node_modules
	-rm -rf src/sdk/pynni/build
	-rm -rf src/sdk/pynni/nni_sdk.egg-info
	-rm -rf src/webui/build
	-rm -rf src/webui/node_modules

# Main targets end

# Helper targets

$(NNI_NODE_TARBALL):
	#$(_INFO) Downloading Node.js $(_END)
	wget https://aka.ms/nni/nodejs-download/$(OS_SPEC) -O $(NNI_NODE_TARBALL)

$(NNI_YARN_TARBALL):
	#$(_INFO) Downloading Yarn $(_END)
	wget https://aka.ms/yarn-download -O $(NNI_YARN_TARBALL)

.PHONY: install-dependencies
install-dependencies: $(NNI_NODE_TARBALL) $(NNI_YARN_TARBALL)
	#$(_INFO) Extracting Node.js $(_END)
	rm -rf $(NNI_NODE_FOLDER)
	mkdir $(NNI_NODE_FOLDER)
	tar -xf $(NNI_NODE_TARBALL) -C $(NNI_NODE_FOLDER) --strip-components 1
	mkdir -p $(BIN_FOLDER)
	rm -f $(NNI_NODE)
	cp $(NNI_NODE_FOLDER)/bin/node $(NNI_NODE)
	
	#$(_INFO) Extracting Yarn $(_END)
	rm -rf $(NNI_YARN_FOLDER)
	mkdir $(NNI_YARN_FOLDER)
	tar -xf $(NNI_YARN_TARBALL) -C $(NNI_YARN_FOLDER) --strip-components 1

.PHONY: install-python-modules
install-python-modules:
	#$(_INFO) Installing Python SDK $(_END)
	cd src/sdk/pynni && $(PIP_INSTALL) $(PIP_MODE) .
	
	#$(_INFO) Installing nnictl $(_END)
	cd tools && $(PIP_INSTALL) $(PIP_MODE) .

.PHONY: dev-install-python-modules
dev-install-python-modules:
	#$(_INFO) Installing Python SDK $(_END)
	cd src/sdk/pynni && $(PIP_INSTALL) $(PIP_MODE) -e .
	
	#$(_INFO) Installing nnictl $(_END)
	cd tools && $(PIP_INSTALL) $(PIP_MODE) -e .

.PHONY: install-node-modules
install-node-modules:
	#$(_INFO) Installing NNI Package $(_END)
	rm -rf $(NNI_PKG_FOLDER)
	cp -r src/nni_manager/dist $(NNI_PKG_FOLDER)
	cp src/nni_manager/package.json $(NNI_PKG_FOLDER)
	$(NNI_YARN) --prod --cwd $(NNI_PKG_FOLDER)
	cp -r src/webui/build $(NNI_PKG_FOLDER)/static

.PHONY: dev-install-node-modules
dev-install-node-modules:
	#$(_INFO) Installing NNI Package $(_END)
	rm -rf $(NNI_PKG_FOLDER)
	ln -sf ${PWD}/src/nni_manager/dist $(NNI_PKG_FOLDER)
	ln -sf ${PWD}/src/nni_manager/node_modules $(NNI_PKG_FOLDER)/node_modules
	ln -sf ${PWD}/src/webui/build $(NNI_PKG_FOLDER)/static

.PHONY: install-scripts
install-scripts:
	mkdir -p $(BASH_COMP_PREFIX)
	install -m644 tools/bash-completion $(BASH_COMP_SCRIPT)

.PHONY: update-bash-config
ifndef _ROOT
update-bash-config:
	#$(_INFO) Updating bash configurations $(_END)
    ifeq (, $(shell echo $$PATH | tr ':' '\n' | grep -x '$(BIN_FOLDER)'))  # $(BIN_FOLDER) not in PATH
	#$(_WARNING) NOTE: adding $(BIN_FOLDER) to PATH in bashrc $(_END)
	echo 'export PATH="$$PATH:$(BIN_FOLDER)"' >> ~/.bashrc
    endif
    ifeq (, $(shell (source ~/.bash_completion ; command -v _nnictl) 2>/dev/null))  # completion not installed
	#$(_WARNING) NOTE: adding $(BASH_COMP_SCRIPT) to ~/.bash_completion $(_END)
	echo '[[ -f $(BASH_COMP_SCRIPT) ]] && source $(BASH_COMP_SCRIPT)' >> ~/.bash_completion
    endif
else
update-bash-config: ;
endif

.PHONY: check-perm
ifdef _ROOT
check-perm:
	#$(_WARNING) Run easy-install as root is not optimal $(_END)
	#$(_WARNING) Suggest run as non-privileged user or manually install instead $(_END)
	#$(_WARNING) Continue easy-install as root? (y/N) $(_END)
	@read CONFIRM && [ "$$CONFIRM" = y ]
else
check-perm: ;
endif

.PHONY: dev-check-perm
ifdef _ROOT
dev-check-perm:
	$(error You should not develop NNI as root)
else
dev-check-perm: ;
endif

# Helper targets end
