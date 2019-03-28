
## Notes

- This example requires tensorflow

- This is an example demonstrating the usage of NNI Network Representation

- Temporarily use `switch.sh` to switch from the subgraph version and the whole graph version

## API Introduction

We design an easy-to-use interface for NAS. It is somewhat like a kind of representation format of **Neural Network Search Space**, which users could easily integrate into their trial's code just like NNI annotation. By doing this, NNI will automatically transform the code to perform NAS.

The first version (might be a naive one) of the API looks like this:
```
"""@nni.architecture
{
    layer_1: {
        layer_choice: [conv3, conv3_sep, conv5, conv5_sep, avg_pool, max_pool],
        input_candidates: [images],
        input_num: 1,
        input_aggregate: None,
        outputs: layer_1_out,
        post_process_outputs: post_process
    },

    layer_2: {
        layer_choice: [conv3, conv3_sep, conv5, conv5_sep, avg_pool, max_pool],
        input_candidates: [layer_1_out],
        input_num: 1,
        input_aggregate: None,
        outputs: layer_2_out,
        post_process_outputs: post_process
    },

    layer_3: {
        layer_choice: [conv3, conv3_sep, conv5, conv5_sep, avg_pool, max_pool],
        input_candidates: [layer_1_out, layer_2_out],
        input_num: 1,
        input_aggregate: None,
        outputs: layer_3_out,
        post_process_outputs: post_process
    }
}"""
```
Every layer is specified by a layer id. For example, `layer_1` in the above code. Following is this layer's information, which is represented like a dict type. Here, variable in `layer_choice`'s list is user-defined function (op they wish to choose), variable in `input_candidates`'s list is user-defined variable (either from the previous layer output or variable defined in code) (which they wish to feed into the above function), and `outputs` is the variable name of the output of this layer (which they could use in the following layer input_candidates or trial's code after this annotation (string) 

NNI will do the following:

1. Select one function from `layer_choice`
2. Select at most `input_num` inputs from `input_candidates`
3. Invoke `input_aggregate` function with the inputs selected above
4. Invoke `layer_choice` with previous return value in step 3
5. Invoke `post_process_output` with previous return value in step 4
6. Assign to a variable whose name is specified by `layer_output`

## Optimize for tensorflow and multi-phase

In general, we have developed one version of the above API. However, in tensorflow, we need to build a graph first and then run the graph. In order to run multiple sub-graph (every sub-graph tuner selected), we need to build multiple graph, which is **an extremely high cost** when every sub-graph just consume relatively little data, say, a mini-batch as it was done in ENAS.

Therefore, we are developing another version for tensorflow. In this version, we will **build the whole graph in a time** using some ugly APIs like tf.case. Latter we just need to change slight to form different sub-graph. Moreover, NNI's multi-phase is used so we only need to build a whole graph in one time and run multiple sub-graph, **increasing efficiency by thousands of times**