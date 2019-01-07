'''@nni.architecture
{
    layer_1: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [images],
        input_num: 1,
        input_aggregate: None,
        outputs: [out],
    },

    layer_2: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [layer_1],
        input_num: 1,
        input_aggregate: None,
        outputs: [out],
    },

    layer_3: {
        layer_choice: [tanh, ReLU, identity, sigmoid],
        input_candidates: [layer_1, layer_2],
        input_num: 1,
        input_aggregate: None,
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


# Further feature
### def tanh(input1, a=0.1) //add hyper-parameter to layer_choice function
### support use index range to select input_candidates //conceive a more elegent design