# 改进 Neural Network Intelligence (NNI)

欢迎！！ 我们非常欢迎贡献者，特别是代码贡献者。

首先，如果有什么不确定的事情，可随时提交问题或拉取请求。 不会有人因此而抱怨。:) 最有可能的是，会有礼貌的请求你修改一些内容。 我们会感激任何形式的贡献，不想用一堆规则来阻止这些贡献。

不管怎样，如果想要更有效的贡献代码，可以阅读以下内容。 本文档包括了所有在贡献中需要注意的要点，会加快合并代码、解决问题的速度。

查看[快速入门](QuickStart.md)来初步了解。

下面是一些简单的贡献指南。

## 提交问题

在提出问题时，请说明以下事项：

* 按照问题模板的内容来填写安装的细节，以便评审者检查。
* 出现问题的场景 (尽量详细，以便重现问题)。
* 错误和日志消息。
* 其它可能有用的细节信息。

## 提交新功能建议

* 在适配使用场景时，总会需要一些新的功能。 可以加入新功能的讨论，也可以直接提交新功能的拉取请求。

* 在自己的 github 账户下 fork 存储库。 在 fork 后， 对于 add, commit, push, 或 squash (如需要) 等改动都需要详细的提交消息。 然后就可以提交拉取请求了。

## 参与源代码和 Bug 修复

拉取请求需要选好正确的标签，表明是 Bug 修复还是功能改进。 所有代码都需要遵循正确的命名约定和代码风格。

参考[如何配置 NNI 的开发环境](./SetupNniDeveloperEnvironment.md)，来安装开发环境。

与[快速入门](QuickStart.md)类似。 其它内容，参考[NNI 文档](http://nni.readthedocs.io)。

## 处理现有问题

查看[问题列表](https://github.com/Microsoft/nni/issues)，找到需要贡献的问题。 可以找找有 'good-first-issue' 或 'help-wanted' 标签的来开始贡献。

修改问题的注释和指派人来表明此问题已经开始跟进。 如果上述问题在一周内没有拉取请求或更新状态，这个问题会重新开放给所有人。 高优先级的 Bug 和回归问题需在一天内响应。

## 代码风格和命名约定

* NNI 遵循 [PEP8](https://www.python.org/dev/peps/pep-0008/) 的 Python 代码命名约定。在提交拉取请求时，请尽量遵循此规范。 可通过`flake8`或`pylint`的提示工具来帮助遵循规范。
* NNI 还遵循 [NumPy Docstring 风格](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy) 的 Python Docstring 命名方案。 Python API 使用了[sphinx.ext.napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) 来[生成文档](Contributing.md#documentation)。
* 有关 docstrings，参考 [numpydoc docstring 指南](https://numpydoc.readthedocs.io/en/latest/format.html) 和 [pandas docstring 指南](https://python-sprints.github.io/pandas/guide/pandas_docstring.html) 
    * 函数的 docstring, **description**, **Parameters**, 以及**Returns**/**Yields** 是必需的。
    * 类的 docstring, **description**, **Attributes** 是必需的。
    * 描述 `dict` 的 docstring 在超参格式描述中多处用到，参考 [RiboKit : 文档标准
    * 写作标准的内部准则](https://ribokit.github.io/docs/text/)

## 文档

文档使用了 [sphinx](http://sphinx-doc.org/) 来生成，支持 [Markdown](https://guides.github.com/features/mastering-markdown/) 和 [reStructuredText](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) 格式。 所有文档都在 [docs/zh_CN](docs) 目录下。

* 在提交文档改动前，请先**在本地生成文档**：`cd docs/zh_CN && make html`，然后，可以在 `docs/zh_CN/_build/html` 目录下找到所有生成的网页。 请认真分析生成日志中的**每个 WARNING**，这非常有可能是或**空连接**或其它问题。

* 需要链接时，尽量使用**相对路径**。 但如果文档是 Markdown 格式的，并且：
    
    * 图片需要通过嵌入的 HTML 语法来格式化，则需要使用绝对链接，如 `https://user-images.githubusercontent.com/44491713/51381727-e3d0f780-1b4f-11e9-96ab-d26b9198ba65.png`。可以通过将图片拖拽到 [Github Issue](https://github.com/Microsoft/nni/issues/new) 框中来生成这样的链接。
    * 如果不能被 sphinx 重新格式化，如源代码等，则需要使用绝对链接。 如果源码连接到本代码库，使用 `https://github.com/Microsoft/nni/tree/master/` 作为根目录 (例如：[mnist.py](https://github.com/Microsoft/nni/blob/master/examples/trials/mnist-tfv1/mnist.py))。