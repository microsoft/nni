# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
FX is a toolkit for developers to use to transform ``nn.Module`` instances. FX consists of three main components, and this pipeline of
components (symbolic tracing -> intermediate representation -> transforms -> Python code generation) constitutes the Python-to-Python
transformation pipeline.

This util consists a **concrete tracer** which extends the **symbolic tracer** in FX. It performs "concrete execution" of the Python code.
Then we can get the **intermediate representation** of ``nn.Module`` instances.

More information about concrete tracing can be found in the :func:`concrete_trace` documentation.
"""
from .concrete_tracer import ConcreteTracer, concrete_trace
from .counter import counter_pass
