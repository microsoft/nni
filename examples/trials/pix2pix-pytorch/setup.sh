#!/bin/bash

# download pix2pix repository
if [ ! -d './pix2pixlib' ] ; then
    git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git pix2pixlib
fi
