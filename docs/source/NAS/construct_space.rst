#####################
Construct Model Space
#####################

NNI provides powerful APIs for users to easily express model space (or search space). First, users can use mutation primitives (e.g., ValueChoice, LayerChoice) to inline a space in their model. Second, NNI provides simple interface for users to customize new mutators for expressing more complicated model spaces. In most cases, the mutation primitives are enough to express users' model spaces.

..  toctree::
    :maxdepth: 1

    Mutation Primitives <MutationPrimitives>
    Customize Mutators <Mutators>
    Hypermodule Lib <Hypermodules>