#####################
Construct Model Space
#####################

NNI provides powerful APIs for users to easily express model space (or search space). First, users can use basic mutation primitives (e.g., ValueChoice, LayerChoice) to embed space in their model. Second, NNI provides popular PyTorch/TensorFlow modules which are a space by themselves. Third, NNI provides simple interface for users to customize new mutators.

..  toctree::
    :maxdepth: 1

    Basic Mutation Primitives <MutationPrimitives>
    Module Space Lib <SupermoduleLib>
    Customize Mutators <Mutators>