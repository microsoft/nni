'''@nni.architecture
{
    block_name: block_1,
    block_inputs: [x_t, h_t-1],
    block_output_aggregate: aggregate_free_outs,
    block_outputs: [h_t],

    layer_1: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [images],
        input_num: 1,
        input_aggregate: null,
        outputs: [out],
    },

    layer_2: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [layer_[1:1]/out],
        input_switch: choose_one,
        outputs: [out],
    },

    layer_3: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [layer_[1:2]/out],
        input_switch: choose_one,
        outputs: [out],
    }
}
'''
# only one input
## get input candidates
layer_1_input_candidate = [graph['layer_1']['input_candidates'][x] for x in nni.get_candidate('layer_1')]
if graph['layer_1']['input_aggregate'] is not None:
    layer_1_aggregated_output = graph['layer_1']['input_aggregate'](*layer_1_input_candidate,)
layer_1_output = graph['layer_1']['layer_choice'][nni.get_layer_choice('layer_1')](layer_1_aggregated_output)