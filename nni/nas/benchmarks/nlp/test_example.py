import json
import pprint
from nni.nas.benchmarks.nlp import query_nlp_trial_stats

a = {"f_input_0":"x","f_input_1":"h_prev_0","f_op":"linear","h_new_0_input_0":"f","h_new_0_op":"activation_tanh"}
for q in query_nlp_trial_stats(arch=a, dataset="ptb"):
    pprint.pprint(q)

print('*' * 100)

b = {"h_new_0_input_0":"node_0","h_new_0_input_1":"node_1","h_new_0_op":"elementwise_sum","node_0_input_0":"x","node_0_input_1":"h_prev_0","node_0_op":"linear","node_1_input_0":"node_0","node_1_op":"activation_tanh"}
for q in query_nlp_trial_stats(arch=b, dataset='wikitext-2', include_intermediates=True):
    pprint.pprint(q['config'])
    pprint.pprint(q['intermediates'][47:49])


