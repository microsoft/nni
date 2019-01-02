# 改进 Neural Network Intelligence (NNI)

欢迎！！ 我们非常欢迎贡献者，特别是代码贡献者。

首先，如果有什么不确定的事情，可随时提交问题或拉取请求。 不会有人因此而抱怨。:) 最有可能的是，会有礼貌的请求你修改一些内容。 我们会感激任何形式的贡献，不想用一堆规则来阻止这些贡献。

不管怎样，如果想要更有效的贡献代码，可以阅读以下内容。 本文档包括了所有在贡献中需要注意的要点，会加快合并代码、解决问题的速度。

查看[快速入门](./GetStarted.md)来初步了解。

下面是一些简单的贡献指南。

## 提出问题

When raising issues, please specify the following:

- Setup details needs to be filled as specified in the issue template clearly for the reviewer to check.
- A scenario where the issue occurred (with details on how to reproduce it).
- Errors and log messages that are displayed by the software.
- Any other details that might be useful.

## Submit Proposals for New Features

- There is always something more that is required, to make it easier to suit your use-cases. Feel free to join the discussion on new features or raise a PR with your proposed change.

- Fork the repository under your own github handle. After cloning the repository. Add, commit, push and sqaush (if necessary) the changes with detailed commit messages to your fork. From where you can proceed to making a pull request.

## Contributing to Source Code and Bug Fixes

Provide PRs with appropriate tags for bug fixes or enhancements to the source code. Do follow the correct naming conventions and code styles when you work on and do try to implement all code reviews along the way.

If you are looking for How to develop and debug the NNI source code, you can refer to [How to set up NNI developer environment doc](./SetupNNIDeveloperEnvironment.md) file in the `docs` folder.

Similarly for [writing trials](./WriteYourTrial.md) or [starting experiments](StartExperiment.md). For everything else, refer [here](https://github.com/Microsoft/nni/tree/master/docs).

## Solve Existing Issues

Head over to [issues](https://github.com/Microsoft/nni/issues) to find issues where help is needed from contributors. You can find issues tagged with 'good-first-issue' or 'help-wanted' to contribute in.

A person looking to contribute can take up an issue by claiming it as a comment/assign their Github ID to it. In case there is no PR or update in progress for a week on the said issue, then the issue reopens for anyone to take up again. We need to consider high priority issues/regressions where response time must be a day or so.

## Code Styles & Naming Conventions

We follow [PEP8](https://www.python.org/dev/peps/pep-0008/) for Python code and naming conventions, do try to adhere to the same when making a pull request or making a change. One can also take the help of linters such as `flake8` or `pylint`