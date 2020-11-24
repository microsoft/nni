## NNI CTL
The NNI CTL module is used to control Neural Network Intelligence, including start a new experiment, stop an experiment and update an experiment etc.
## Environment

```
Ubuntu 16.04 or other Linux OS
python >= 3.6
```

## Installation

1. Enter tools directory

1. Use pip to install packages
    * Install for current user:

      ```bash
      python3 -m pip install --user -e .
      ```

    * Install for all users:

      ```bash
      python3 -m pip install -e .
      ```

1. Change the mode of nnictl file

    ```bash
    chmod +x ./nnictl
    ```

1. Add nnictl to your PATH system environment variable.

    * You could use `export` command to set PATH variable temporary.

          export PATH={your nnictl path}:$PATH

    * Or you could edit your `/etc/profile` file.

      ```txt
      1.sudo vim /etc/profile

      2.At the end of the file, add

            export PATH={your nnictl path}:$PATH

      save and exit.

      3.source /etc/profile
      ```

## To start using NNI CTL

please reference to the [NNI CTL document].

[NNI CTL document]: ../docs/en_US/Nnictl.md
