def max_pool(k):
    pass
h_conv1=1
"""@nni.variable(nni.choice(2,3,5,7),name=conv_size)"""
conv_size = 5
"""@nni.variable(nni.choice('2',3,5*6,{1:2, '3':4},[1,2,3]),name=abc)"""
abc = 5
"""@nni.function_choice(max_pool(h_conv1), avg_pool(h_conv2,h_conv3), name=max_pool)"""
h_pool1 = max_pool(h_conv1)
"""@nni.function_choice(max_poo(h_conv1), 2 * 3 + 4, lambda x: 1+x, name=max_poo)"""
h_pool2 = max_poo(h_conv1)
test_acc=1
'''@nni.report_intermediate_result(test_acc)'''
test_acc=2
'''@nni.report_final_result(test_acc)'''
