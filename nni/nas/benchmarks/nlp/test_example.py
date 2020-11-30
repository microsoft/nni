import json
import pprint
from nni.nas.benchmarks.nlp import query_nlp_trial_stats

# a = {"f_input_0":"x","f_input_1":"h_prev_0","f_op":"linear","h_new_0_input_0":"f","h_new_0_op":"activation_tanh"}
# a= {"h_new_0_input_0":"node_5","h_new_0_input_1":"node_12","h_new_0_input_2":"node_10","h_new_0_op":"linear","node_0_input_0":"h_prev_0","node_0_input_1":"x","node_0_op":"linear","node_10_input_0":"node_9","node_10_op":"activation_tanh","node_11_input_0":"node_10","node_11_input_1":"x","node_11_op":"linear","node_12_input_0":"node_11","node_12_op":"activation_sigm","node_1_input_0":"node_0","node_1_op":"activation_leaky_relu","node_2_input_0":"x","node_2_input_1":"node_1","node_2_input_2":"h_prev_0","node_2_op":"linear","node_3_input_0":"node_2","node_3_op":"activation_leaky_relu","node_4_input_0":"h_prev_0","node_4_input_1":"x","node_4_op":"linear","node_5_input_0":"node_4","node_5_op":"activation_sigm","node_7_input_0":"node_1","node_7_input_1":"node_3","node_7_op":"linear","node_8_input_0":"node_7","node_8_op":"activation_leaky_relu","node_9_input_0":"node_5","node_9_input_1":"node_8","node_9_op":"linear"}
a= {"h_new_0_input_0":"node_3","h_new_0_input_1":"x","h_new_0_op":"linear","node_2_input_0":"x","node_2_input_1":"h_prev_0","node_2_op":"linear","node_3_input_0":"node_2","node_3_op":"activation_leaky_relu"}
for q in query_nlp_trial_stats(arch=a, dataset="ptb"):
    pprint.pprint(q)

# print('*' * 100)



# b = {"h_new_0_input_0":"node_0","h_new_0_input_1":"node_1","h_new_0_op":"elementwise_sum","node_0_input_0":"x","node_0_input_1":"h_prev_0","node_0_op":"linear","node_1_input_0":"node_0","node_1_op":"activation_tanh"}
# for q in query_nlp_trial_stats(arch=b, dataset='wikitext-2', include_intermediates=True):
#     pprint.pprint(q['config'])
    # pprint.pprint(q['intermediates'][47:49])


