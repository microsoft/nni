with open(FLAGS.data_path, "rb") as finp:
    x_train, x_valid, x_test, _, _ = pickle.load(finp,encoding="latin1")
 
x_train, y_train, num_train_batches = ptb_input_producer(x_train, batch_size, bptt_steps)
y_train = tf.reshape(y_train, [batch_size * bptt_steps])
 
// build training model
w_emb = tf.get_variable("w", [vocab_size, lstm_hidden_size])
embedding = tf.nn.embedding_lookup(w_emb, x_train)
...
embedding *= e_mask
 
def linearize(inputs):
    if len(inputs) == 2:
        ht = tf.matmul(tf.concat([inputs[0] * x_mask, inputs[1] * s_mask], axis=1), new_weights)
    elif len(inputs) == 1:
        ht = tf.matmul(inputs[0] * x_mask, new_weights)
    ht = batch_norm(ht, true)
    h, t = tf.split(ht, 2, axis=1)
    return h, t
    
def weighted_sum(h, t, prev_s):
    t = tf.sigmoid(t)
    s = prev_s + t * (h - prev_s)
    return s
    
def aggregate_free_outs(free_outs):
    out = tf.reduce_mean(free_outs, axis = 0)
    out.set_shape([batch_size, self.lstm_hidden_size])
    out = batch_norm(out, true)
    return out
 
def body(step, prev_h, all_h):
    next_h = []
    for layer_id, p_h in enumerate(zip(prev_h)):
        if layer_id == 0:
            inputs = embedding[:, step, :]
        else:
            inputs = next_h[-1]

        // IR (Domain-specific language)
        // N = 6
        {
            block_name: block_1
            block_inputs: [inputs, p_h]
            block_post_process: aggregate_free_outs(free_outs)
            block_outputs: [h_t]
 
            layer_1: {
                input_candidates: [inputs, p_h]
                input_number: 2
                pre_process: linearize(inputs)
                layer_choice: [tanh(pre_process[0]), ReLU(pre_process[0]), identity(pre_process[0]), sigmoid(pre_process[0])]
                post_process: weighted_sum(layer_choice[0], pre_process[1], inputs[1])
                outputs: [out]
            }
 
            layer_2: {
                input_candidates: [layer_1/outputs/out]
                input_number: 1
                pre_process: linearize(inputs)
                layer_choice: [tanh(pre_process[0]), ReLU(pre_process[0]), identity(pre_process[0]), sigmoid(pre_process[0])]
                post_process: weighted_sum(layer_choice[0], pre_process[1], inputs[0])
                outputs: [out]
            }
 
            layer_3: {
                input_candidates: [layer_[1:2]/outputs/out]
                input_number: 1
                pre_process: linearize
                layer_choice: [tanh(pre_process[0]), ReLU(pre_process[0]), identity(pre_process[0]), sigmoid(pre_process[0])]
                post_process: weighted_sum(layer_choice[0], pre_process[1], inputs[0])
                outputs: [out]
            }
 
            layer_4: {
                input_candidates: [layer_[1:3]/outputs/out]
                input_number: 1
                pre_process: linearize
                layer_choice: [tanh(pre_process[0]), ReLU(pre_process[0]), identity(pre_process[0]), sigmoid(pre_process[0])]
                post_process: weighted_sum(layer_choice[0], pre_process[1], inputs[0])
                outputs: [out]
            }
 
            layer_5: {
                input_candidates: [layer_[1:4]/outputs/out]
                input_number: 1
                pre_process: linearize
                layer_choice: [tanh(pre_process[0]), ReLU(pre_process[0]), identity(pre_process[0]), sigmoid(pre_process[0])]
                post_process: weighted_sum(layer_choice[0], pre_process[1], inputs[0])
                outputs: [out]
            }
 
            layer_6: {
                input_candidates: [layer_[1:5]/outputs/out]
                input_number: 1
                pre_process: linearize
                layer_choice: [tanh(pre_process[0]), ReLU(pre_process[0]), identity(pre_process[0]), sigmoid(pre_process[0])]
                post_process: weighted_sum(layer_choice[0], pre_process[1], inputs[0])
                outputs: [out]
            }
        }
        next_h.append(block_1/block_outputs/h_t)
    out_h = next_h[-1]
    out_h *= o_mask
    all_h = all_h.write(step, out_h)
    return step + 1, next_h, all_h
    
loop_vars = [tf.constant(0, dtype=tf.int32), start_h, all_h]
loop_outputs = tf.while_loop(condition, body, loop_vars, back_prop=True)
