import nni

def max_pool(k):
    pass


h_conv1 = 1
conv_size = nni.choice(2, 3, 5, 7, name='conv_size')
h_pool1 = nni.function_choice(lambda : max_pool(h_conv1), lambda : avg_pool
    (h_conv2, h_conv3), name='max_pool')
test_acc = 1
nni.report_intermediate_result(test_acc)
test_acc = 2
nni.report_final_result(test_acc)
