#!/bin/bash

# download and preprocess the Stanford Dogs dataset

mkdir -p data/stanford-dogs

# download raw data (images, annotations, and train-test split)
cd data/stanford-dogs

if [ ! -d './Images' ] ; then
  if [ ! -f 'images.tar' ] ; then
    wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
  fi
  tar -xvf images.tar
fi

if [ ! -d './Annotation' ] ; then
  if [ ! -f 'annotation.tar' ] ; then
    wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
  fi
  tar -xvf annotation.tar
fi

if [ ! -f 'lists.tar' ] ; then
  wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
fi
tar -xvf lists.tar

cd ../..

# preprocess: train-valid-test splitting and image cropping
python preprocess.py
