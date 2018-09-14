Scripts to create deb package for Ubuntu.

## Usage

Install bulid tools:

    # apt install dpkg-dev devscripts debhelper

Install yarn, node, serve, and required Python packages in your favored way.

This should have been done if you are developing NNI.

Build deb package (on real Ubuntu machine):

    $ debuild -us -uc

If you are using WSL, use this command instead:

    $ debuild -us -uc -rfakeroot-tcp

## Issues

* The `copyright` file needs review.
* Whenever adding a new Python dependency to NNI, `postinst` must be updated manually.
* `python3-json-tricks` is omitted in dependency vector because it is only available in Ubuntu 18.04+.
* The post-install script (which invokes pip to install optional dependencies) will skip Python packages installed locally.
* Need to add optional package metadata.
* Warnings of `debuild` are ignored.
* Need test case.
* These scripts do not take advantage of `debuild`'s Python helper utilities.
