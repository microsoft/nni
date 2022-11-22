# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

from nni.compression.pytorch.speedup.jit_translate import parse_aten_schema_version_1_8_x, schema_fix_dict, special_treat_dict

def parse_aten_schema_origin(schema: str):
    if schema in schema_fix_dict:
        schema = schema_fix_dict[schema]

    positional_num = 0
    keyword_list = list()
    special_treat = dict() # for dtype and memory_format trans now

    for arg in torch._C.parse_schema(schema).arguments:
        if torch.__version__ < '1.9.0' or not arg.kwarg_only:
            key = positional_num
            positional_num += 1
        else:
            key = arg.name
            keyword_list.append(key)

        if arg.name in special_treat_dict:
            if key not in special_treat:
                special_treat[key] = [special_treat_dict[arg.name]]
            else:
                special_treat[key].append(special_treat_dict[arg.name])

    return positional_num, keyword_list, special_treat

class SchemaParserTestCase(unittest.TestCase):
    def test_diff_manual_parser(self):
        all_schema_list = (str(i) for i in torch._C._jit_get_all_schemas())
        for schema in all_schema_list:
            if not schema.startswith('aten::'):
                continue
            if torch.__version__ < '1.9.0' and '*,' in schema:
                continue
            positional_num_origin, keyword_list_origin, special_treat_origin = parse_aten_schema_origin(schema)
            positional_num_manual, keyword_list_manual, special_treat_manual = parse_aten_schema_version_1_8_x(schema)
            
            assert positional_num_origin == positional_num_manual
            assert keyword_list_origin == keyword_list_manual
            assert special_treat_origin == special_treat_manual

if __name__ == '__main__':
    unittest.main()
