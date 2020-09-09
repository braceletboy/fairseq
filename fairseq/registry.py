# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
@file: registry.py

[FILEDONE] This file contains a function for setting up a registry. The fariseq
module maintains various registries each for a different class of modules (eg:
OPTIMIZER_REGISTRY, MODEL_REGISTRY, MONOTONIC_ATTENTION_REGISTRY etc). The
registry is a dictionary that maps a name to the an instance.

@readby: rukmangadh.sai@nobroker.in
'''


import argparse

'''
GLOBALVAR: REGISTRIES

[DONE] This global variable maintains a registry of all the registries.

@readby: rukmangadh.sai@nobroker.in
'''
REGISTRIES = {}


'''
FUNCTION: setup_registry()

ARGUMENTS:
    registry_name: Every registry has a name associated with it. This is that
    name.
    base_class: Every instance in a registry is an object of classes inherited
    from a certain base class. This is that base class.
    default: Each instance in the registry is registered with a name. This is
    the name of the default name.
    required: If it's required/compulsory for the user to specify which
    instance in the REGISTRY (that's being setup) should be used.

[DONE] This function is useful in setting up a new registry.

@readby: rukmangadh.sai@nobroker.in
'''


def setup_registry(
    registry_name: str,
    base_class=None,
    default=None,
    required=False,
):
    assert registry_name.startswith('--')
    registry_name = registry_name[2:].replace('-', '_')

    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()

    # maintain a registry of all registries
    if registry_name in REGISTRIES:
        return  # registry already exists
    REGISTRIES[registry_name] = {
        'registry': REGISTRY,
        'default': default,
    }

    '''
    FUNCTION: build_x()

    ARGUMENTS:
        args: The command line arguments

    [DONE] This is closure function that maintains a pointer to the variables
    REGISTRY and required even after we have gone out of the scope of the outer
    function setup_registry().

    When this closure function is used, it calls the builder corresponding to
    the *choice* inside the REGISTRY and returns it's return values.

    @readby: rukmangadh.sai@nobroker.in
    '''
    def build_x(args, *extra_args, **extra_kwargs):
        choice = getattr(args, registry_name, None)
        if choice is None:
            if required:
                raise ValueError('--{} is required!'.format(registry_name))
            return None
        cls = REGISTRY[choice]
        if hasattr(cls, 'build_' + registry_name):
            builder = getattr(cls, 'build_' + registry_name)
        else:
            builder = cls
        '''
        [GOTO set_defaults() in fairseq/registry.py]

        @readby: rukmangadh.sai@nobroker.in
        '''
        set_defaults(args, cls)
        return builder(args, *extra_args, **extra_kwargs)

    '''
    FUNCTION: register_x()

    ARGUMENTS:
        name: The name with which to register the instance.

    [DONE] This is a decorator that maintains a pointer to the variables
    REGISTRY and base_class even after we have gone out of the scope of the
    outer function setup_registry()

    This decorator is used for registering classes into the REGISTRY.

    @readby: rukmangadh.sai@nobroker.in
    '''
    def register_x(name):

        '''
        FUNCTION: register_x_cls()

        ARGUMENTS:
            cls: The class that is being registered

        [DONE] This is the inner function of the register_x() decorator that
        actually registers the class. The outer function is required because we
        want to be able to customize the variable *name* while registering each
        class.

        @readby: rukmangadh.sai@nobroker.in
        '''
        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError('Cannot register duplicate {} ({})'.format(registry_name, name))
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    'Cannot register {} with duplicate class name ({})'.format(
                        registry_name, cls.__name__,
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError('{} must extend {}'.format(cls.__name__, base_class.__name__))
            REGISTRY[name] = cls
            REGISTRY_CLASS_NAMES.add(cls.__name__)
            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY


'''
FUNCTION: set_defaults()

ARGUMENTS:
    args: The command line arguments passed (this is the object obtained after
    doing parser.parse_args())
    cls: The class which has the 'add_args' function and whose additional
    arguments require to be set to their default values.

[DONE] Set defaults to the arguments in modules. Modules can have additional
arguments that are specific for that module. These are defined in that module
file. These arguments are added through the *add_args* static method in those
modules. See the documentation for how new modules are added to get a better
hang of this add_args() method.
Eg: https://fairseq.readthedocs.io/en/latest/models.html?highlight=add_args#adding-new-models

An observation that I have made is that this function is used for setting of
defaults for some internal modules that are not user defined like base

@readby: rukmangadh.sai@nobroker.in
'''


def set_defaults(args, cls):
    """Helper to set default arguments based on *add_args*."""
    if not hasattr(cls, 'add_args'):
        return
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, allow_abbrev=False)
    cls.add_args(parser)
    # copied from argparse.py:
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, default_value)
