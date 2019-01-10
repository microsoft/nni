# Setting variables
SHELL := /bin/bash
PIP_INSTALL := python3 -m pip install --no-cache-dir
PIP_UNINSTALL := python3 -m pip uninstall

## Colorful output
_INFO := $(shell echo -e '\033[1;36m')
_WARNING := $(shell echo -e '\033[1;33m')
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

## Install directories
ROOT_FOLDER ?= $(shell python3 -c 'import site; from pathlib import Path; print(Path(site.getsitepackages()[0]).parents[2])')
IS_SYS_PYTHON ?= $(shell [[ $(ROOT_FOLDER) == /usr* || $(ROOT_FOLDER) == /Library* ]] && echo TRUE || echo FALSE)

ifeq ($(shell id -u), 0)  # is root
    _ROOT := 1
    BASH_COMP_PREFIX ?= /usr/share/bash-completion/completions
else  # is normal user
    ifeq (TRUE, $(IS_SYS_PYTHON))
        ROOT_FOLDER := $(shell python3 -c 'import site; from pathlib import Path; print(Path(site.getusersitepackages()).parents[2])')
    endif
    ifndef VIRTUAL_ENV
        ifeq (, $(shell echo $$PATH | grep 'conda'))
            PIP_MODE ?= --user
        endif
    endif
    BASH_COMP_PREFIX ?= ${HOME}/.bash_completion.d
endif
BASH_COMP_SCRIPT := $(BASH_COMP_PREFIX)/nnictl

NNI_INSTALL_PATH ?= $(INSTALL_PREFIX)/nni

BIN_FOLDER ?= $(ROOT_FOLDER)/bin
NNI_PKG_FOLDER ?= $(ROOT_FOLDER)/nni

## Dependency information
NNI_DEPENDENCY_FOLDER = /tmp/$(USER)
$(shell mkdir -p $(NNI_DEPENDENCY_FOLDER))
NNI_NODE_TARBALL ?= $(NNI_DEPENDENCY_FOLDER)/nni-node-$(OS_SPEC)-x64.tar.xz
NNI_NODE_FOLDER = $(NNI_DEPENDENCY_FOLDER)/nni-node-$(OS_SPEC)-x64
NNI_NODE ?= $(BIN_FOLDER)/node
NNI_YARN_TARBALL ?= $(NNI_DEPENDENCY_FOLDER)/nni-yarn.tar.gz
NNI_YARN_FOLDER ?= $(NNI_DEPENDENCY_FOLDER)/nni-yarn
NNI_YARN := PATH=$(BIN_FOLDER):$${PATH} $(NNI_YARN_FOLDER)/bin/yarn

## Version number
NNI_VERSION_VALUE = $(shell git describe --tags)
NNI_VERSION_TEMPLATE = 999.0.0-developing

# Main targets

.PHONY: build
build:
	#$(_INFO) Building NNI Manager $(_END)
	cd src/nni_manager && $(NNI_YARN) && $(NNI_YARN) build
	#$(_INFO) Building WebUI $(_END)
	cd src/webui && $(NNI_YARN) && $(NNI_YARN) build

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
	cd src/sdk/pynni && sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' setup.py && $(PIP_INSTALL) $(PIP_MODE) .
	
	#$(_INFO) Installing nnictl $(_END)
	cd tools && sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' setup.py && $(PIP_INSTALL) $(PIP_MODE) .

.PHONY: dev-install-python-modules
dev-install-python-modules:
	#$(_INFO) Installing Python SDK $(_END)
	cd src/sdk/pynni && sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' setup.py && $(PIP_INSTALL) $(PIP_MODE) -e .
	
	#$(_INFO) Installing nnictl $(_END)
	cd tools && sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' setup.py && $(PIP_INSTALL) $(PIP_MODE) -e .

.PHONY: install-node-modules
install-node-modules:
	#$(_INFO) Installing NNI Package $(_END)
	rm -rf $(NNI_PKG_FOLDER)
	cp -r src/nni_manager/dist $(NNI_PKG_FOLDER)
	cp src/nni_manager/package.json $(NNI_PKG_FOLDER)
	sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' $(NNI_PKG_FOLDER)/package.json
	$(NNI_YARN) --prod --cwd $(NNI_PKG_FOLDER)
	cp -r src/webui/build $(NNI_PKG_FOLDER)/static

.PHONY: dev-install-node-modules
dev-install-node-modules:
	#$(_INFO) Installing NNI Package $(_END)
	rm -rf $(NNI_PKG_FOLDER)
	ln -sf ${PWD}/src/nni_manager/dist $(NNI_PKG_FOLDER)
	cp src/nni_manager/package.json $(NNI_PKG_FOLDER)
	sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' $(NNI_PKG_FOLDER)/package.json
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
