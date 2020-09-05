#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json


def load_from_json_file(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data


def save_to_json_file(data, file_name):
    with open(file_name, "w") as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4)
