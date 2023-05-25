# WebUI

WebUI is built by using [React](https://reactjs.org/docs/getting-started.html) and [fluentui](https://developer.microsoft.com/en-us/fluentui#/controls/web).


## Development

* Please refer the [installation doc](https://github.com/microsoft/nni#installation) to run an experiment.

* Use this command in `webui/ts` directory when you change webui code. And then refresh website to see latest pages.
    ```bash
    yarn build
    ```

## PR

* WebUI uses [eslint](https://eslint.org/docs/user-guide/getting-started) and [prettier](https://prettier.io/docs/en/index.html) to format code. You could use the command `yarn sanity-check` to check the code error status. And use `yarn eslint-fix` could modifiy the most code style error before you send PR. Also Please use `yarn stylelint --fix` to format `css and scss` files.

* You could send the PR if `yarn release` gets successful build after formatting code.