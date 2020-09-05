#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import json


def _list_to_dict(model_list):
    assert isinstance(model_list, list)
    names = [x["name"] for x in model_list]
    assert len(set(names)) == len(model_list), f"Name not unique {names}"

    ret = {val["name"]: val for val in model_list}
    return ret


def load_model_info(file_name, to_dict=True):
    with open(file_name, "r") as fp:
        ret = json.load(fp)
    if to_dict:
        ret = _list_to_dict(ret)
    return ret


def load_model_info_all(folder_path):
    paths = glob.glob(f"{folder_path}/*.json")
    ret = []
    for path in paths:
        cur = load_model_info(path, to_dict=False)
        ret += cur
    ret = _list_to_dict(ret)
    return ret


def get_model_info_folder(name):
    import pkg_resources

    sub_folder = "model_info"
    if name is not None:
        sub_folder += f"/{name}"
    ret = pkg_resources.resource_filename(__name__, sub_folder)
    return ret
