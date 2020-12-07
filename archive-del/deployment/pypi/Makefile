CWD := $(PWD)/

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
	OS_SPEC := linux
	WHEEL_SPEC := manylinux1_x86_64
	NODE_URL := https://nodejs.org/dist/v10.22.1/node-v10.22.1-linux-x64.tar.xz
else ifeq ($(UNAME_S), Darwin)
	OS_SPEC := darwin
	WHEEL_SPEC := macosx_10_9_x86_64
	NODE_URL := https://nodejs.org/dist/v10.22.1/node-v10.22.1-darwin-x64.tar.xz
else
	$(error platform $(UNAME_S) not supported)
endif

TIME_STAMP = $(shell date -u "+%y%m%d%H%M")
NNI_VERSION_VALUE = $(shell git describe --tags --abbrev=0)
NNI_VERSION_VALUE := $(NNI_VERSION_VALUE:v%=%)

# To include time stamp in version value, run:
# make version_ts=true build
ifeq ($(version_ts), true)
	NNI_VERSION_VALUE := $(NNI_VERSION_VALUE).$(TIME_STAMP)
endif
NNI_VERSION_TEMPLATE = 999.0.0-developing
NNI_YARN_TARBALL ?= $(CWD)nni-yarn.tar.gz
NNI_YARN_FOLDER ?= $(CWD)nni-yarn
NNI_YARN := PATH=$(CWD)node-$(OS_SPEC)-x64/bin:$${PATH} $(NNI_YARN_FOLDER)/bin/yarn
.PHONY: build
build:
	# Building version $(NNI_VERSION_VALUE)
	python3 -m pip install --user --upgrade setuptools wheel
	wget -q $(NODE_URL) -O $(CWD)node-$(OS_SPEC)-x64.tar.xz
	rm -rf $(CWD)node-$(OS_SPEC)-x64
	mkdir $(CWD)node-$(OS_SPEC)-x64
	tar xf $(CWD)node-$(OS_SPEC)-x64.tar.xz -C node-$(OS_SPEC)-x64 --strip-components 1
	wget -q https://github.com/yarnpkg/yarn/releases/download/v1.22.5/yarn-v1.22.5.tar.gz -O $(NNI_YARN_TARBALL)
	rm -rf $(NNI_YARN_FOLDER)
	mkdir $(NNI_YARN_FOLDER)
	tar -xf $(NNI_YARN_TARBALL) -C $(NNI_YARN_FOLDER) --strip-components 1
	cd $(CWD)../../src/nni_manager && $(NNI_YARN) && $(NNI_YARN) build
	cd $(CWD)../../src/webui && $(NNI_YARN) && $(NNI_YARN) build
	cd $(CWD)../../src/nasui && $(NNI_YARN) && $(NNI_YARN) build
	rm -rf $(CWD)nni
	cp -r $(CWD)../../src/nni_manager/dist $(CWD)nni
	cp -r $(CWD)../../src/nni_manager/config $(CWD)nni
	cp -r $(CWD)../../src/webui/build $(CWD)nni/static
	mkdir -p $(CWD)nni/nasui/build
	cp -r $(CWD)../../src/nasui/build/. $(CWD)nni/nasui/build
	cp $(CWD)../../src/nasui/server.js $(CWD)nni/nasui
	cp $(CWD)../../src/nni_manager/package.json $(CWD)nni
	sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' $(CWD)nni/package.json
	cd $(CWD)nni && $(NNI_YARN) --prod
	sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' $(CWD)../../src/sdk/pynni/nni/__init__.py
	cd $(CWD) && sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' setup.py && python3 setup.py bdist_wheel -p $(WHEEL_SPEC)
	cd $(CWD)

.PHONY: upload
upload:
	python3 -m pip install --user --upgrade twine
	python3 -m twine upload dist/*

.PHONY: clean
clean:
	rm -rf $(CWD)build
	rm -rf $(CWD)dist
	rm -rf $(CWD)nni
	rm -rf $(CWD)nni.egg-info
	rm -rf $(CWD)node-$(OS_SPEC)-x64
