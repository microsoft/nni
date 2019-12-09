##############
NAS Algorithms
##############

Automatic neural architecture search is taking an increasingly important role on finding better models.
Recent research works have proved the feasibility of automatic NAS, and also found some models that could beat manually designed and tuned models.
Some of representative works are NASNet, ENAS, DARTS, Network Morphism, and Evolution. There are new innovations keeping emerging.

However, it takes great efforts to implement NAS algorithms, and it is hard to reuse code base of existing algorithms in new one.
To facilitate NAS innovations (e.g., design and implement new NAS models, compare different NAS models side-by-side),
an easy-to-use and flexible programming interface is crucial.

With this motivation, our ambition is to provide a unified architecture in NNI,
to accelerate innovations on NAS, and apply state-of-art algorithms on real world problems faster.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Overview <NAS/Overview>
    NAS Interface <NAS/NasInterface>
    ENAS <NAS/ENAS>
    DARTS <NAS/DARTS>
    P-DARTS <NAS/Overview>
