# Tuning Transformer with Retiarii

This demo is adapted from PyTorch Transformer tutorial.
Here, we show how we use functions provided by retiarii to tune Transformer's hyper-parameters, in order to achieve better performance.
This demo is tested with PyTorch 1.9, torchtext == 0.10, and nni == 2.4.
Please change the configurations (starting on line 196) accordingly and then run: `python retiarii_transformer_demo.py`

We use a built-in dataset provided by torchtext, WikiText-2, to evaluate Transformer on language modeling. We tune two hyper-parameters: the number of encoder layers (`n_layer`) whose default value in the original paper is 6, and the dropout rate shared by all encoder layers (`p_dropout`) whose default value is 0.1. We report validation perplexity as metric (the lower is better).

We first tune one hyper-parameter with another fixed to the default value. The results are:
![separate](https://user-images.githubusercontent.com/22978940/136937420-80aecee9-43cc-4f8d-b282-18aec0ad3929.png)

And then we tune these two hyper-parameters jointly. The results are:
<p align="center">
  <img src="https://user-images.githubusercontent.com/22978940/136937807-342fde98-6498-4cdd-abdd-4633fd15b7dc.png" width="700">
</p>

As we can observe, we have found better hyper-parameters (`n_layer = 8`, `p_dropout = 0.2`) than default values. 

