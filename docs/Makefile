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

# Build message catelogs for translation
.PHONY: i18n
i18n:
	sphinx-build -b getpartialtext source build/i18n
	sphinx-intl update -p build/i18n -d source/locales -l zh

.PHONY: clean
clean:
	rm -rf build
	rm -rf source/reference/_modules
