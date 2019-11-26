#################
Model Compression
#################

NNI provides an easy-to-use toolkit to help user design and use compression algorithms.
It supports Tensorflow and PyTorch with unified interface.
For users to compress their models, they only need to add several lines in their code.
There are some popular model compression algorithms built-in in NNI.
Users could further use NNI's auto tuning power to find the best compressed model,
which is detailed in Auto Model Compression.
On the other hand, users could easily customize their new compression algorithms using NNI's interface.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Overview <Compressor/Overview>
    Level Pruner <Compressor/Pruner>
    AGP Pruner <Compressor/Pruner>
    L1Filter Pruner <Compressor/L1FilterPruner>
    Slim Pruner <Compressor/SlimPruner>
    Lottery Ticket Pruner <Compressor/LotteryTicketHypothesis>
    FPGM Pruner <Compressor/Pruner>
    Naive Quantizer <Compressor/Quantizer>
    QAT Quantizer <Compressor/Quantizer>
    DoReFa Quantizer <Compressor/Quantizer>
    Automatic Model Compression <Compressor/AutoCompression>
