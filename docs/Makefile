.PHONY: all
all: en zh

.PHONY: en
en:
	## English ##
	sphinx-build -T source build

.PHONY: zh
zh:
	## Chinese ##
	sphinx-build -T -D language=zh source build_zh

.PHONY: clean
clean:
	rm -rf build build_zh
