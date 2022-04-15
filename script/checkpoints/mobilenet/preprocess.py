# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import xml.etree.ElementTree
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import io


ROOT_DIR = './data/stanford-dogs/'
NUM_CATEGORIES = 120
OUT_IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42                        # for splitting train and validation
TRAIN_RATIO = 0.9                       # train / (train + validation)


def get_bounding_box(annotation_file):
    """
    Parse the annotation file and returns the bounding box information
    Parameters
    ----------
    annotation_file: path to the annotation XML file

    Returns
    -------
    A dict containing bounding box information
    """
    ret = {}
    xml_root = xml.etree.ElementTree.parse(annotation_file).getroot()
    bounding_box = xml_root.findall('object')[0].findall('bndbox')[0]
    ret['X_min'] = int(bounding_box.findall('xmin')[0].text)
    ret['X_max'] = int(bounding_box.findall('xmax')[0].text)
    ret['Y_min'] = int(bounding_box.findall('ymin')[0].text)
    ret['Y_max'] = int(bounding_box.findall('ymax')[0].text)

    return ret


def main(root_dir):
    try:
        os.mkdir(root_dir + 'Processed')
        os.mkdir(root_dir + 'Processed/train')
        os.mkdir(root_dir + 'Processed/valid')
        os.mkdir(root_dir + 'Processed/test')
    except:
        print('Directory already exists. Nothing done.')
        exit()
        
    # load train test splits
    train_metadata = io.loadmat(root_dir + 'train_list.mat')
    train_valid_file_list = [x[0][0] for x in train_metadata['file_list']]
    train_valid_annotation_list = [x[0][0] for x in train_metadata['annotation_list']]
    train_valid_labels = [x[0] - 1 for x in train_metadata['labels']]
    train_valid_lists = [x for x in zip(train_valid_file_list, train_valid_annotation_list, train_valid_labels)]
    train_lists, valid_lists = train_test_split(train_valid_lists, train_size=TRAIN_RATIO, random_state=RANDOM_SEED)
    train_file_list, train_annotation_list, train_labels = zip(*train_lists)
    valid_file_list, valid_annotation_list, valid_labels = zip(*valid_lists)

    test_metadata = io.loadmat(root_dir + 'test_list.mat')
    test_file_list = [x[0][0] for x in test_metadata['file_list']]
    test_annotation_list = [x[0][0] for x in test_metadata['annotation_list']]
    test_labels = [x[0] - 1 for x in test_metadata['labels']]

    label2idx = {}
    for split, file_list, annotation_list, labels in zip(['train', 'valid', 'test'],
                                                         [train_file_list, valid_file_list, test_file_list],
                                                         [train_annotation_list, valid_annotation_list, test_annotation_list],
                                                         [train_labels, valid_labels, test_labels]):
        print('Preprocessing {} set: {} cases'.format(split, len(file_list)))
        for cur_file, cur_annotation, cur_label in zip(file_list, annotation_list, labels):
            label_name = cur_file.split('/')[0].split('-')[-1].lower()
            if label_name not in label2idx:
                label2idx[label_name] = cur_label
            image = Image.open(root_dir + '/Images/' + cur_file)

            # cropping and reshape
            annotation_file = root_dir + '/Annotation/' + cur_annotation
            bounding_box = get_bounding_box(annotation_file)
            image = image.crop([bounding_box['X_min'], bounding_box['Y_min'],
                                bounding_box['X_max'], bounding_box['Y_max']])
            image = image.convert('RGB')
            image = image.resize(OUT_IMAGE_SIZE)

            # Normalize and save the instance
            X = np.array(image)
            X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))          # normalize each channel separately

            # image.save(root_dir + 'Processed/' + split + '/' + image_name)
            np.save(root_dir + 'Processed/' + split + '/' + cur_file.split('/')[-1].replace('.jpg', '.npy'),
                    {'input': X, 'label': cur_label})

    # save mapping from label name to index to a dict
    with open(ROOT_DIR + '/category_dict.tsv', 'w') as dict_f:
        final_dict_list = sorted(list(label2idx.items()), key=(lambda x: x[-1]))
        for label, index in final_dict_list:
            dict_f.write('{}\t{}\n'.format(index, label))
        print(final_dict_list)


if __name__ == '__main__':
    main(ROOT_DIR)
