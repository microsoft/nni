# NNI Annotation Introduction 

For good user experience and reduce user effort, we need to design a good annotation grammar.

If users use NNI system, they only need to:

 1. Annotation variable in code as:

    '''@nni.variable(nni.choice(2,3,5,7),name=self.conv_size)'''

 2. Annotation intermediate in code as:

    '''@nni.report_intermediate_result(test_acc)'''

 3. Annotation output in code as:

    '''@nni.report_final_result(test_acc)'''

 4. Annotation `function_choice` in code as:

    '''@nni.function_choice(max_pool(h_conv1, self.pool_size),avg_pool(h_conv1, self.pool_size),name=max_pool)'''

In this way, they can easily implement automatic tuning on NNI.

For `@nni.variable`, `nni.choice` is the type of search space and there are 10 types to express your search space as follows:

 1. `@nni.variable(nni.choice(option1,option2,...,optionN),name=variable)`  
    Which means the variable value is one of the options, which should be a list The elements of options can themselves be stochastic expressions

 2. `@nni.variable(nni.randint(upper),name=variable)`  
    Which means the variable value is a random integer in the range [0, upper).

 3. `@nni.variable(nni.uniform(low, high),name=variable)`  
    Which means the variable value is a value uniformly between low and high.

 4. `@nni.variable(nni.quniform(low, high, q),name=variable)`  
    Which means the variable value is a value like round(uniform(low, high) / q) * q

 5. `@nni.variable(nni.loguniform(low, high),name=variable)`  
    Which means the variable value is a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed.

 6. `@nni.variable(nni.qloguniform(low, high, q),name=variable)`  
    Which means the variable value is a value like round(exp(uniform(low, high)) / q) * q

 7. `@nni.variable(nni.normal(label, mu, sigma),name=variable)`  
    Which means the variable value is a real value that's normally-distributed with mean mu and standard deviation sigma.

 8. `@nni.variable(nni.qnormal(label, mu, sigma, q),name=variable)`  
    Which means the variable value is a value like round(normal(mu, sigma) / q) * q

 9. `@nni.variable(nni.lognormal(label, mu, sigma),name=variable)`  
    Which means the variable value is a value drawn according to exp(normal(mu, sigma))

 10. `@nni.variable(nni.qlognormal(label, mu, sigma, q),name=variable)`  
    Which means the variable value is a value like round(exp(normal(mu, sigma)) / q) * q
