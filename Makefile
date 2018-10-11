# Setting variables

SHELL := /bin/bash
PIP_INSTALL := python3 -m pip install
PIP_UNINSTALL := python3 -m pip uninstall

## Colorful output
_INFO := $(shell echo -e '\e[1;36m')
_WARNING := $(shell echo -e '\e[1;33m')
_END := $(shell echo -e '\e[0m')


## Install directories
ifeq ($(shell id -u), 0)  # is root
    _ROOT := 1
    BIN_PATH ?= /usr/bin
    INSTALL_PREFIX ?= /usr/share
    EXAMPLES_PATH ?= $(INSTALL_PREFIX)/nni/examples
    BASH_COMP_SCRIPT ?= /usr/share/bash-completion/completions/nnictl
else  # is normal user
    BIN_PATH ?= ${HOME}/.local/bin
    INSTALL_PREFIX ?= ${HOME}/.local
    EXAMPLES_PATH ?= ${HOME}/nni/examples
    ifndef VIRTUAL_ENV
        PIP_MODE ?= --user
    endif
    BASH_COMP_SCRIPT ?= ${HOME}/.bash_completion.d/nnictl
endif

## Dependency information
NODE_VERSION ?= v10.10.0
NODE_TARBALL ?= node-$(NODE_VERSION)-linux-x64.tar.xz
NODE_PATH ?= $(INSTALL_PREFIX)/nni/node

YARN_VERSION ?= v1.9.4
YARN_TARBALL ?= yarn-$(YARN_VERSION).tar.gz
YARN_PATH ?= /tmp/nni-yarn

## Check if dependencies have been installed globally
ifeq (, $(shell command -v node 2>/dev/null))
    $(info $(_INFO) Node.js not found $(_END))
    _MISS_DEPS := 1  # node not found
else
    _VER := $(shell node --version)
    _NEWER := $(shell echo -e "$(NODE_VERSION)\n$(_VER)" | sort -Vr | head -n 1)
    ifneq ($(_VER), $(_NEWER))
        $(info $(_INFO) Node.js version not match $(_END))
        _MISS_DEPS := 1  # node outdated
    endif
endif
ifeq (, $(shell command -v yarnpkg 2>/dev/null))
    $(info $(_INFO) Yarn not found $(_END))
    _MISS_DEPS := 1  # yarn not found
endif

ifdef _MISS_DEPS
    $(info $(_INFO) Missing dependencies, use local toolchain $(_END))
    NODE := $(NODE_PATH)/bin/node
    YARN := PATH=$${PATH}:$(NODE_PATH)/bin $(YARN_PATH)/bin/yarn
else
    $(info $(_INFO) All dependencies found, use global toolchain $(_END))
    NODE := node
    YARN := yarnpkg
endif


# Setting variables end


# Main targets

.PHONY: build
build:
	#$(_INFO) Building NNI Manager $(_END)
	cd src/nni_manager && $(YARN) && $(YARN) build
	
	#$(_INFO) Building WebUI $(_END)
	cd src/webui && $(YARN) && $(YARN) build
	
	#$(_INFO) Building Python SDK $(_END)
	cd src/sdk/pynni && python3 setup.py build
	
	#$(_INFO) Building nnictl $(_END)
	cd tools && python3 setup.py build

# Standard installation target
# Must be invoked after building
.PHONY: install
install: install-python-modules
install: install-node-modules
install: install-scripts
install: install-examples
install:
	#$(_INFO) Complete! You may want to add $(BIN_PATH) to your PATH environment $(_END)


# Target for remote machine workers
# Only installs core SDK module
.PHONY: remote-machine-install
remote-machine-install:
	cd src/sdk/pynni && python3 setup.py install $(PIP_MODE)


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


# Target for setup.py
# Do not invoke this manually
.PHONY: pip-install
pip-install: install-dependencies
pip-install: build
pip-install: install-node-modules
pip-install: install-scripts
pip-install: install-examples
pip-install: update-bash-config


# Target for NNI developers
# Creates symlinks instead of copying files
.PHONY: dev-install
dev-install: check-dev-env
dev-install: install-dev-modules
dev-install: install-scripts
dev-install:
	#$(_INFO) Complete! You may want to add $(BIN_PATH) to your PATH environment $(_END)


.PHONY: uninstall
uninstall:
	-$(PIP_UNINSTALL) -y nni
	-$(PIP_UNINSTALL) -y nnictl
	-rm -rf $(INSTALL_PREFIX)/nni
	-rm -f $(BIN_PATH)/nnimanager
	-rm -f $(BIN_PATH)/nnictl
	-rm -f $(BASH_COMP_SCRIPT)
	-[ $(EXAMPLES_PATH) = ${PWD}/examples ] || rm -rf $(EXAMPLES_PATH)

# Main targets end


# Helper targets

$(NODE_TARBALL):
	#$(_INFO) Downloading Node.js $(_END)
	wget https://nodejs.org/dist/$(NODE_VERSION)/$(NODE_TARBALL)

$(YARN_TARBALL):
	#$(_INFO) Downloading Yarn $(_END)
	wget https://github.com/yarnpkg/yarn/releases/download/$(YARN_VERSION)/$(YARN_TARBALL)

.PHONY: intall-dependencies
install-dependencies: $(NODE_TARBALL) $(YARN_TARBALL)
	#$(_INFO) Cleaning $(_END)
	rm -rf $(NODE_PATH)
	rm -rf $(YARN_PATH)
	mkdir -p $(NODE_PATH)
	mkdir -p $(YARN_PATH)
	
	#$(_INFO) Extracting Node.js $(_END)
	tar -xf $(NODE_TARBALL)
	mv -fT node-$(NODE_VERSION)-linux-x64 $(NODE_PATH)
	
	#$(_INFO) Extracting Yarn $(_END)
	tar -xf $(YARN_TARBALL)
	mv -fT yarn-$(YARN_VERSION) $(YARN_PATH)

.PHONY: install-python-modules
install-python-modules:
	#$(_INFO) Installing Python SDK $(_END)
	cd src/sdk/pynni && $(PIP_INSTALL) $(PIP_MODE) .
	
	#$(_INFO) Installing nnictl $(_END)
	cd tools && $(PIP_INSTALL) $(PIP_MODE) .

.PHONY: install-node-modules
install-node-modules:
	mkdir -p $(INSTALL_PREFIX)/nni
	rm -rf src/nni_manager/dist/node_modules
	
	#$(_INFO) Installing NNI Manager $(_END)
	cp -rT src/nni_manager/dist $(INSTALL_PREFIX)/nni/nni_manager
	cp -rT src/nni_manager/node_modules $(INSTALL_PREFIX)/nni/nni_manager/node_modules
	
	#$(_INFO) Installing WebUI $(_END)
	cp -rT src/webui/build $(INSTALL_PREFIX)/nni/nni_manager/static


.PHONY: install-dev-modules
install-dev-modules:
	#$(_INFO) Installing Python SDK $(_END)
	cd src/sdk/pynni && $(PIP_INSTALL) $(PIP_MODE) -e .
	
	#$(_INFO) Installing nnictl $(_END)
	cd tools && $(PIP_INSTALL) $(PIP_MODE) -e .

	mkdir -p $(INSTALL_PREFIX)/nni
	
	#$(_INFO) Installing NNI Manager $(_END)
	ln -sf ${PWD}/src/nni_manager/dist $(INSTALL_PREFIX)/nni/nni_manager
	ln -sf ${PWD}/src/nni_manager/node_modules $(INSTALL_PREFIX)/nni/nni_manager/node_modules
	
	#$(_INFO) Installing WebUI $(_END)
	ln -sf ${PWD}/src/webui/build $(INSTALL_PREFIX)/nni/nni_manager/static


.PHONY: install-scripts
install-scripts:
	mkdir -p $(BIN_PATH)
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnimanager
	echo 'cd $(INSTALL_PREFIX)/nni/nni_manager' >> $(BIN_PATH)/nnimanager
	echo '$(NODE) main.js $$@' >> $(BIN_PATH)/nnimanager
	chmod +x $(BIN_PATH)/nnimanager
	
	echo '#!/bin/sh' > $(BIN_PATH)/nnictl
	echo 'NNI_MANAGER=$(BIN_PATH)/nnimanager \' >> $(BIN_PATH)/nnictl
	echo 'python3 -m nnicmd.nnictl $$@' >> $(BIN_PATH)/nnictl
	chmod +x $(BIN_PATH)/nnictl
	
	install -Dm644 tools/bash-completion $(BASH_COMP_SCRIPT)


.PHONY: install-examples
install-examples:
	mkdir -p $(EXAMPLES_PATH)
	[ $(EXAMPLES_PATH) = ${PWD}/examples ] || cp -rT examples $(EXAMPLES_PATH)


.PHONY: update-bash-config
ifndef _ROOT
update-bash-config:
	#$(_INFO) Updating bash configurations $(_END)
    ifeq (, $(shell echo $$PATH | tr ':' '\n' | grep -x '$(BIN_PATH)'))  # $(BIN_PATH) not in PATH
	#$(_WARNING) NOTE: adding $(BIN_PATH) to PATH in bashrc $(_END)
	echo 'export PATH="$$PATH:$(BIN_PATH)"' >> ~/.bashrc
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


.PHONY: check-dev-env
check-dev-env:
	#$(_INFO) Checking developing environment... $(_END)
ifdef _ROOT
	$(error You should not develop NNI as root)
endif
ifdef _MISS_DEPS
#	$(error Please install Node.js and Yarn to develop NNI)
endif
	#$(_INFO) Pass! $(_END)

# Helper targets end
