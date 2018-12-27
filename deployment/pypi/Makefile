CWD := $(PWD)/

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
	OS_SPEC := linux
	WHEEL_SPEC := manylinux1_x86_64
else ifeq ($(UNAME_S), Darwin)
	OS_SPEC := darwin
	WHEEL_SPEC := macosx_10_9_x86_64
else 
	$(error platform $(UNAME_S) not supported)
endif

NNI_VERSION_VALUE = $(shell git describe --tags)
NNI_VERSION_TEMPLATE = 999.0.0-developing

.PHONY: build
build:
	python3 -m pip install --user --upgrade setuptools wheel
	wget https://aka.ms/nni/nodejs-download/$(OS_SPEC) -O $(CWD)node-$(OS_SPEC)-x64.tar.xz
	rm -rf $(CWD)node-$(OS_SPEC)-x64
	mkdir $(CWD)node-$(OS_SPEC)-x64
	tar xf $(CWD)node-$(OS_SPEC)-x64.tar.xz -C node-$(OS_SPEC)-x64 --strip-components 1
	cd $(CWD)../../src/nni_manager && yarn && yarn build
	cd $(CWD)../../src/webui && yarn && yarn build
	rm -rf $(CWD)nni
	cp -r $(CWD)../../src/nni_manager/dist $(CWD)nni
	cp -r $(CWD)../../src/webui/build $(CWD)nni/static
	cp $(CWD)../../src/nni_manager/package.json $(CWD)nni
	sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' $(CWD)nni/package.json
	cd $(CWD)nni && yarn --prod
	cd $(CWD) && sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' setup.py && python3 setup.py bdist_wheel -p $(WHEEL_SPEC)
	cd $(CWD)../../src/sdk/pynni && sed -ie 's/$(NNI_VERSION_TEMPLATE)/$(NNI_VERSION_VALUE)/' setup.py && python3 setup.py bdist_wheel
	cp -r $(CWD)../../src/sdk/pynni/dist/*.whl $(CWD)dist
	cd $(CWD)

.PHONY: upload
upload:
	python3 -m pip install --user --upgrade twine
	python3 -m twine upload dist/*

.PHONY: clean
clean:
	rm -rf $(CWD)../../src/sdk/pynni/dist
	rm -rf $(CWD)build
	rm -rf $(CWD)dist
	rm -rf $(CWD)nni
	rm -rf $(CWD)nni.egg-info
	rm -rf $(CWD)node-$(OS_SPEC)-x64