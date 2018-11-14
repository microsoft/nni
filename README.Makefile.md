# Makefile and Installation Setup

NNI uses GNU make for building and installing.

The `Makefile` offers standard targets `build`, `install`, and `uninstall`, as well as alternative installation targets for different setup:

* `easy-install`: target for non-expert users, which handles everything automatically;
* `dev-easy-install`: target for developer users, which handles everything automatically;
* `install`: target for NNI normal users, which installs NNI by copying files;
* `dev-install`: target for NNI contributors, which installs NNI as symlinks instead of copying files;
* `pip-install`: target in favor of `setup.py`;

The targets will be detailed later.

## Dependencies

NNI requires at least Node.js, Yarn, and pip to build, while TypeScript is also recommended.

NNI requires Node.js, and all dependency libraries to run.
Required Node.js libraries (including TypeScript) can be installed by Yarn, and required Python libraries can be installed by setuptools or PIP.

For NNI *users*, `make install-dependencies` can be used to install Node.js and Yarn.
This will install Node.js to NNI's installation directory, and install Yarn to `/tmp/nni-yarn`.
This target requires wget to work.

For NNI *developers*, it is recommended to install Node.js and Yarn manually.
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

In some configuration, NNI will also install Node.js to `/usr/share/nni`.

All directories mentioned above are configurable. See next section for details.

### Configuration

The `Makefile` uses environment variables to override default settings.

Available variables are listed below:

| Name               | Description                                             | Default for normal user           | Default for root                                |
|--------------------|---------------------------------------------------------|-----------------------------------|-------------------------------------------------|
| `BIN_PATH`         | Path for executables                                    | `~/.local/bin`                    | `/usr/bin`                                      |
| `INSTALL_PREFIX`   | Path for Node.js modules (a suffix `nni` will be added) | `~/.local`                        | `/usr/share`                                    |
| `BASH_COMP_SCRIPT` | Path of bash completion script                          | `~/.bash_completion.d/nnictl`     | `/usr/share/bash-completion/completions/nnictl` |
| `PIP_MODE`         | Arguments for `python3 setup.py install`                | `--user` if `VIRTUAL_ENV` not set | (empty)                                         |
| `NODE_PATH`        | Path to install Node.js runtime                         | `$INSTALL_PREFIX/nni/node`        | `$INSTALL_PREFIX/nni/node`                      |
| `YARN_PATH`        | Path to install Yarn                                    | `/tmp/nni-yarn`                   | `/tmp/nni-yarn`                                 |
| `NODE`             | Node.js command                                         | see source file                   | see source file                                 |
| `YARN`             | Yarn command                                            | see source file                   | see source file                                 |

Note that these variables will influence installation destination as well as generated `nnictl` and `nnimanager` scripts.
If the path to copy files is different from where they will run (e.g. when creating a distro package), please generate `nnictl` and `nnimanager` manually.

### Targets

The workflow of each installation targets is listed below:

| Target                   | Workflow                                                                   |
|--------------------------|----------------------------------------------------------------------------|
| `easy-install`           | Install dependencies, build, install NNI, and edit `~/.bashrc`             |
| `dev-easy-install`       | Install dependencies, build, install NNI as symlinks, and edit `~/.bashrc` |
| `install`                | Install Python packages, Node.js modules, NNI scripts, and examples        |
| `dev-install`            | Install Python and Node.js modules as symlinks, then install scripts       |
| `pip-install`            | Install dependencies, build, install NNI excluding Python packages         |

## TODO

* `clean` target
* `test` target
* `lint` target
* Test cases for each target
* Review variables
