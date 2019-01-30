# Makefile and Installation Setup

NNI uses GNU make for building and installing.

The `Makefile` offers standard targets `build`, `install`, and `uninstall`, as well as alternative installation targets for different setup:

* `easy-install`: target for non-expert users, which handles everything automatically;
* `pip-install`: target in favor of `setup.py`;
* `dev-install`: target for NNI contributors, which installs NNI as symlinks instead of copying files;
* `remote-machine-install`: target that only installs core Python library for remote machine workers.

The targets will be detailed later.

## Dependencies

NNI requires at least Node.js, Yarn, and setuptools to build, while PIP and TypeScript are also recommended.

NNI requires Node.js, serve, and all dependency libraries to run.
Required Node.js libraries (including TypeScript) can be installed by Yarn, and required Python libraries can be installed by setuptools or PIP.

For NNI *users*, `make install-dependencies` can be used to install Node.js, Yarn, and serve.
This will install Node.js and serve to NNI's installation directory, and install Yarn to `/tmp/nni-yarn`.
This target requires wget to work.

For NNI *developers*, it is recommended to install Node.js, Yarn, and serve manually.
See their official sites for installation guide.

## Building NNI

Simply run `make` when dependencies are ready.

## Installation

### Directory Hierarchy

The main parts of NNI project consist of two Node.js modules (`nni_manager`, `webui`) and two Python packages (`nni`, `nnictl`).

By default the Node.js modules are installed to `/usr/share/nni` for all users or installed to `~/.local/nni` for current user.

The Python packages are installed with setuptools and therefore the location depends on Python configuration.
When install as non-priviledged user and virtualenv is not detected, `--user` flag will be used.

In addition, `nnictl` offers a bash completion scripts, which will be installed to `/usr/share/bash-completion/completions` or `~/.bash_completion.d`.

In some configuration, NNI will also install Node.js and the serve module to `/usr/share/nni`.

All directories mentioned above are configurable. See next section for details.

### Configuration

The `Makefile` uses environment variables to override default settings.

Available variables are listed below:

| Name               | Description                                             | Default for normal user           | Default for root                                |
|--------------------|---------------------------------------------------------|-----------------------------------|-------------------------------------------------|
| `BIN_PATH`         | Path for executables                                    | `~/.local/bin`                    | `/usr/bin`                                      |
| `INSTALL_PREFIX`   | Path for Node.js modules (a suffix `nni` will be added) | `~/.local`                        | `/usr/share`                                    |
| `EXAMPLES_PATH`    | Path for NNI examples                                   | `~/nni/examples`                  | `$INSTALL_PREFIX/nni/examples`                  |
| `BASH_COMP_SCRIPT` | Path of bash completion script                          | `~/.bash_completion.d/nnictl`     | `/usr/share/bash-completion/completions/nnictl` |
| `PIP_MODE`         | Arguments for `python3 setup.py install`                | `--user` if `VIRTUAL_ENV` not set | (empty)                                         |
| `NODE_PATH`        | Path to install Node.js runtime                         | `$INSTALL_PREFIX/nni/node`        | `$INSTALL_PREFIX/nni/node`                      |
| `SERVE_PATH`       | Path to install serve package                           | `$INSTALL_PREFIX/nni/serve`       | `$INSTALL_PREFIX/nni/serve`                     |
| `YARN_PATH`        | Path to install Yarn                                    | `/tmp/nni-yarn`                   | `/tmp/nni-yarn`                                 |
| `NODE`             | Node.js command                                         | see source file                   | see source file                                 |
| `SERVE`            | serve command                                           | see source file                   | see source file                                 |
| `YARN`             | Yarn command                                            | see source file                   | see source file                                 |

Note that these variables will influence installation destination as well as generated `nnictl` and `nnimanager` scripts.
If the path to copy files is different from where they will run (e.g. when creating a distro package), please generate `nnictl` and `nnimanager` manually.

### Targets

The workflow of each installation targets is listed below:

| Target                   | Workflow                                                             |
|--------------------------|----------------------------------------------------------------------|
| `install`                | Install Python packages, Node.js modules, NNI scripts, and examples  |
| `easy-install`           | Install dependencies, build, install NNI, and edit `~/.bashrc`       |
| `pip-install`            | Install dependencies, build, install NNI excluding Python packages   |
| `dev-install`            | Install Python and Node.js modules as symlinks, then install scripts |
| `remote-machine-install` | Install `nni` Python package                                         |

## TODO

* `clean` target
* `test` target
* `lint` target
* Exclude tuners and their dependencies from `remote-machine-install`
* Test cases for each target
* Review variables
