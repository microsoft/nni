# NNI Annotation 介绍

为了获得良好的用户体验并减少用户负担，NNI 设计了通过注释来使用的语法。

使用 NNI 时，只需要:

1. 在超参变量前加上如下标记：
    
    '''@nni.variable(nni.choice(2,3,5,7),name=self.conv_size)'''

2. 在中间结果前加上：
    
    '''@nni.report_intermediate_result(test_acc)'''

3. 在输出结果前加上：
    
    '''@nni.report_final_result(test_acc)'''

4. 在代码中使用函数 `function_choice`：
    
    '''@nni.function_choice(max_pool(h_conv1, self.pool_size),avg_pool(h_conv1, self.pool_size),name=max_pool)'''

通过这种方法，能够轻松的在 NNI 中实现自动调参。

`@nni.variable`, `nni.choice` 为搜索空间的类型，通过以下 10 种方法来定义搜索空间：

1. `@nni.variable(nni.choice(option1,option2,...,optionN),name=variable)`  
    变量值是选项中的一种，这些变量可以是任意的表达式。

2. `@nni.variable(nni.randint(upper),name=variable)`  
    变量可以是范围 [0, upper) 中的任意整数。

3. `@nni.variable(nni.uniform(low, high),name=variable)`  
    变量值会是 low 和 high 之间均匀分布的某个值。

4. `@nni.variable(nni.quniform(low, high, q),name=variable)`  
    变量值会是 low 和 high 之间均匀分布的某个值，公式为：round(uniform(low, high) / q) * q

5. `@nni.variable(nni.loguniform(low, high),name=variable)`  
    变量值是 exp(uniform(low, high)) 的点，数值以对数均匀分布。

6. `@nni.variable(nni.qloguniform(low, high, q),name=variable)`  
    变量值会是 low 和 high 之间均匀分布的某个值，公式为：round(exp(uniform(low, high)) / q) * q

7. `@nni.variable(nni.normal(label, mu, sigma),name=variable)`  
    变量值为正态分布的实数值，平均值为 mu，标准方差为 sigma。

8. `@nni.variable(nni.qnormal(label, mu, sigma, q),name=variable)`  
    变量值分布的公式为： round(normal(mu, sigma) / q) * q

9. `@nni.variable(nni.lognormal(label, mu, sigma),name=variable)`  
    变量值分布的公式为： exp(normal(mu, sigma))

10. `@nni.variable(nni.qlognormal(label, mu, sigma, q),name=variable)`  
    变量值分布的公式为： round(exp(normal(mu, sigma)) / q) * q