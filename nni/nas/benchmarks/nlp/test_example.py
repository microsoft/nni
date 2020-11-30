import json
import pprint
from nni.nas.benchmarks.nlp import query_nlp_trial_stats

a = {'node_0_op': 'linear', 'node_0_input_0': 'x', 'node_0_input_1': 'h_prev_0', 'node_1_op': 'activation_sigm', 'node_1_input_0': 'node_0', 'node_4_op': 'linear', 'node_4_input_0': 'x', 'node_4_input_1': 'node_1', 'node_4_input_2': 'h_prev_0', 'node_5_op': 'activation_leaky_relu', 'node_5_input_0': 'node_4', 'node_8_op': 'linear', 'node_8_input_0': 'node_5', 'node_8_input_1': 'x', 'h_new_0_op': 'activation_sigm', 'h_new_0_input_0': 'node_8'}
b = "{'h_new_0_input_0': 'node_3', 'h_new_0_input_1': 'node_2', 'h_new_0_input_2': 'node_1', 'h_new_0_op': 'blend', 'node_0_input_0': 'x', 'node_0_input_1': 'h_prev_0', 'node_0_op': 'linear','node_1_input_0': 'node_0', 'node_1_op': 'activation_tanh', 'node_2_input_0': 'h_prev_0', 'node_2_input_1': 'node_1', 'node_2_input_2': 'x', 'node_2_op': 'linear', 'node_3_input_0': 'node_2', 'node_3_op': 'activation_leaky_relu'}"
c = {'h_new_0': {'input': ['node_3', 'node_2', 'node_1'], 'op': 'blend'}, 'node_0': {'input': ['x', 'h_prev_0'], 'op': 'linear'}, 'node_1': {'input': ['node_0'], 'op': 'activation_tanh'}, 'node_2': {'input': ['h_prev_0', 'node_1', 'x'], 'op': 'linear'}, 'node_3': {'input': ['node_2'], 'op': 'activation_leaky_relu'}}
for i in query_nlp_trial_stats(arch=c, dataset="ptb"):
    pprint.pprint(i)

d = {"h_new_0_input_0":"node_0","h_new_0_input_1":"node_1","h_new_0_op":"elementwise_sum","node_0_input_0":"x","node_0_input_1":"h_prev_0","node_0_op":"linear","node_1_input_0":"node_0","node_1_op":"activation_tanh"}
for i in query_nlp_trial_stats(arch=d, dataset='wikitext-2', include_intermediates=True):
    pprint.pprint(i['intermediates'][47:49])


