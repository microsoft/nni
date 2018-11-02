CWD := $(PWD)/

.PHONY: build
build:
	python3 -m pip install --user --upgrade setuptools wheel
	wget https://aka.ms/nodejs-download -O $(CWD)node-linux-x64.tar.xz
	rm -rf $(CWD)node-linux-x64
	mkdir $(CWD)node-linux-x64
	tar xf $(CWD)node-linux-x64.tar.xz -C node-linux-x64 --strip-components 1
	cd $(CWD)../../src/nni_manager && yarn && yarn build
	cd $(CWD)../../src/webui && yarn && yarn build
	rm -rf $(CWD)nni_pkg
	cp -r $(CWD)../../src/nni_manager/dist $(CWD)nni_pkg
	cp -r $(CWD)../../src/webui/build $(CWD)nni_pkg/static
	cp $(CWD)../../src/nni_manager/package.json $(CWD)nni_pkg
	cd $(CWD)nni_pkg && yarn --prod
	rm -rf $(CWD)nnicmd
	rm -rf $(CWD)nni_annotation
	cp -r $(CWD)../../tools/nnicmd $(CWD)nnicmd
	cp -r $(CWD)../../tools/nni_annotation $(CWD)nni_annotation
	cp -r $(CWD)../../tools/trial_tool $(CWD)trial_tool
	cd $(CWD) && python3 setup.py bdist_wheel
	cd $(CWD)../../src/sdk/pynni && python3 setup.py bdist_wheel
	cp -r $(CWD)../../src/sdk/pynni/dist/*.whl $(CWD)dist
	cd $(CWD)

.PHONY: upload
upload:
	python3 -m pip install --user --upgrade twine
	python3 -m twine upload dist/*