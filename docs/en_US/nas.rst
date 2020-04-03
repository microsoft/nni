##########################
Neural Architecture Search
##########################

Automatic neural architecture search is taking an increasingly important role on finding better models.
Recent research works have proved the feasibility of automatic NAS, and also found some models that could beat manually tuned models.
Some of representative works are NASNet, ENAS, DARTS, Network Morphism, and Evolution. Moreover, new innovations keep emerging.

However, it takes great efforts to implement NAS algorithms, and it is hard to reuse code base of existing algorithms in a new one.
To facilitate NAS innovations (e.g., design and implement new NAS models, compare different NAS models side-by-side),
an easy-to-use and flexible programming interface is crucial.

Therefore, we provide a unified interface for NAS,
to accelerate innovations on NAS, and apply state-of-art algorithms on real world problems faster.
For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Overview <NAS/Overview>
    Quick Start <NAS/QuickStart>
    Tutorial <NAS/NasGuide>
    ENAS <NAS/ENAS>
    DARTS <NAS/DARTS>
    P-DARTS <NAS/PDARTS>
    SPOS <NAS/SPOS>
    CDARTS <NAS/CDARTS>
    ProxylessNAS <NAS/Proxylessnas>
    TextNAS <NAS/TextNAS>
    Customize a NAS Algorithm <NAS/Advanced>
    API Reference <NAS/NasReference>
