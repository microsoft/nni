.. bb39a6ac0ae1f5554bc38604c77fb616

#####################
构建模型空间
#####################

NNI为用户提供了强大的API，以方便表达模型空间（或搜索空间）。 首先，用户可以使用 mutation 原语（如 ValueChoice、LayerChoice）在他们的模型中内联一个空间。 其次，NNI为用户提供了简单的接口，可以定制新的 mutators 来表达更复杂的模型空间。 在大多数情况下，mutation 原语足以表达用户的模型空间。

..  toctree::
    :maxdepth: 1

    mutation 原语 <MutationPrimitives>
    定制 mutator <Mutators>
    Hypermodule Lib <Hypermodules>