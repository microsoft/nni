def get_total_num_weights(model, op_types=['default']):
        '''
        calculate the total number of weights

        Returns
        -------
        int
            total weights of all the op considered
        '''
        num_weights = 0
        for _, module in model.named_modules():
            if module == model:
                continue
            if 'default' in op_types or type(module).__name__ in op_types:
                num_weights += module.weight.data.numel()
        return num_weights