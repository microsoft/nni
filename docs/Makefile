.PHONY: all
all: en zh

.PHONY: en
en:
	## English ##
	sphinx-build -T source build/html

.PHONY: zh
zh:
	## Chinese ##
	sphinx-build -T -D language=zh source build/html_zh

.PHONY: clean
clean:
	rm -rf build
	rm -rf source/reference/_modules
