# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
@file: hubconf.py

[FILEDONE] Torch Hub is a pre-trained model repository maintained by pytorch.
You can write a hubconf.py file in the github repository that contains the
model being developed and publish your model at [https://pytorch.org/hub/]

Refer to [https://pytorch.org/docs/stable/hub.html] for learning why this file
is written this way.

@readby: rukmangadh.sai@nobroker.in
'''


import functools

from fairseq.hub_utils import BPEHubInterface as bpe  # noqa
from fairseq.hub_utils import TokenizerHubInterface as tokenizer  # noqa
from fairseq.models import MODEL_REGISTRY


dependencies = [
    'numpy',
    'regex',
    'requests',
    'torch',
]


'''
I think the torch.hub module executes the whole of hubconf.py as a script. So,
the following code gets exected and the cython components in the repository get
built before making the models available through torch.hub.list()

@readby: rukmangadh.sai@nobroker.in
'''

# torch.hub doesn't build Cython components, so if they are not found then try
# to build them here
try:
    import fairseq.data.token_block_utils_fast
except (ImportError, ModuleNotFoundError):
    try:
        import cython
        import os
        from setuptools import sandbox
        sandbox.run_setup(
            os.path.join(os.path.dirname(__file__), 'setup.py'),
            ['build_ext', '--inplace'],
        )
    except (ImportError, ModuleNotFoundError):
        print(
            'Unable to build Cython components. Please make sure Cython is '
            'installed if the torch.hub model you are loading depends on it.'
        )


'''
The following just adds the elements in MODEL_REGISTRY to the global variables
dictionary, maintained by python for this script. I think this is another way,
apart from the ways described at [https://pytorch.org/docs/stable/hub.html], to
expose the functions so that torch.hub.list() can show them when queried.

@readby: rukmangadh.sai@nobroker.in
'''

for _model_type, _cls in MODEL_REGISTRY.items():
    for model_name in _cls.hub_models().keys():
        globals()[model_name] = functools.partial(
            _cls.from_pretrained,
            model_name,
        )
    # to simplify the interface we only expose named models
    # globals()[_model_type] = _cls.from_pretrained
