NNI is under development to support JupyterLab.
You can install this extension to preview the feature.

Currently you can view NNI web UI inside JupyterLab.

## Install ##

To preview the extension, you need to have `nni` and `jupyterlab` installed at first:

```
$ pip install nni jupyterlab
```

Then run following command to register extension:

```
$ nnictl jupyter-extension install
```

It does not have prompt message. Exit without error means success.

## Run ##

For now, the extension does not support creating experiment, so you have to create one with nnictl:

```
$ nnictl create --config <experiment-directory>/config.yml
```

And you need to launch JupyterLab:

```
$ jupyter lab --ip=0.0.0.0
```

Following JupyterLab's guide to open its web page, you should find an NNI icon.
Click the icon and it will open NNI web UI for your running experiment.

## Uninstall ##

To uninstall (or more accurately, unregister) the extension, run following command:

```
$ nnictl jupyter-extension uninstall
```

## Known Issues ##

The JupyterLab extension is under development and there are many issues need to fix before public announcement:

  * Clicking a link in experiment management page will open it outside JupyterLab. To fix it will need modify in web UI.
  * Downloading log file might not work.
  * Post requests (update experiment config) might not work.
