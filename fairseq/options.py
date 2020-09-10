# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
@file: options.py

[FILENOTDONE] This file as the name suggests, is for defining and then
configuring the command line options/flags provided by the fairseq module.

@readby: rukmangadh.sai@nobroker.in
'''


import argparse
import sys
from typing import Callable, List, Optional

import torch

from fairseq import utils
from fairseq.data.indexed_dataset import get_available_dataset_impl


def get_preprocessing_parser(default_task="translation"):
    parser = get_parser("Preprocessing", default_task)
    add_preprocess_args(parser)
    return parser


'''
FUNCTION: get_training_parser()

ARGUMENTS:
    default_task: The default task to use for the --task flag

[DONE] This function creates and returns the parser required for training.

@readby: rukmangadh.sai@nobroker.in
'''


def get_training_parser(default_task="translation"):
    '''
    [GOTO get_parser() IN fairseq/options.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    parser = get_parser("Trainer", default_task)
    '''
    [GOTO add_dataset_args() IN fariseq/options.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    add_dataset_args(parser, train=True)
    '''
    [GOTO add_distributed_training_args() IN fariseq/options.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    add_distributed_training_args(parser)
    '''
    [GOTO add_model_args() IN fairseq/options.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    add_model_args(parser)
    '''
    [GOTO add_optimization_args() IN fairseq/options.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    add_optimization_args(parser)
    '''
    [GOTO add_checkpoint_args() IN fariseq/options.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    add_checkpoint_args(parser)
    return parser


def get_generation_parser(interactive=False, default_task="translation"):
    parser = get_parser("Generation", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_generation_args(parser)
    if interactive:
        add_interactive_args(parser)
    return parser


def get_interactive_generation_parser(default_task="translation"):
    return get_generation_parser(interactive=True, default_task=default_task)


def get_eval_lm_parser(default_task="language_modeling"):
    parser = get_parser("Evaluate Language Model", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_eval_lm_args(parser)
    return parser


def get_validation_parser(default_task=None):
    parser = get_parser("Validation", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser, default_world_size=1)
    group = parser.add_argument_group("Evaluation")
    add_common_eval_args(group)
    return parser


def csv_str_list(x):
    return x.split(',')


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_str_dict(x, type=dict):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    return x


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


'''
FUNCTION: parse_args_and_arch()

ARGUMENTS:
    parser: The parser for parsing command line arguments
    input_args:
    parse_known: If True, doesn't raise error if unknown args are given
    suppress_defaults:
    modify_parser:

[PARTIALLYDONE] This function add the module specific arguments at various
steps and then finally parses them, it finally returns the parsed arguments
(and the left over flags if we set parse_known to True). It also set's the
architecture specific flags to the corresponding values at the end.

@readby: rukmangadh.sai@nobroker.in
'''


def parse_args_and_arch(
    parser: argparse.ArgumentParser,
    input_args: List[str] = None,
    parse_known: bool = False,
    suppress_defaults: bool = False,
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """
    Args:
        parser (ArgumentParser): the parser
        input_args (List[str]): strings to parse, defaults to sys.argv
        parse_known (bool): only parse known arguments, similar to
            `ArgumentParser.parse_known_args`
        suppress_defaults (bool): parse while ignoring all default values
        modify_parser (Optional[Callable[[ArgumentParser], None]]):
            function to modify the parser, e.g., to set default values
    """
    if suppress_defaults:
        # Parse args without any default values. This requires us to parse
        # twice, once to identify all the necessary task/model args, and a second
        # time with all defaults set to None.
        args = parse_args_and_arch(
            parser,
            input_args=input_args,
            parse_known=parse_known,
            suppress_defaults=False,
        )
        suppressed_parser = argparse.ArgumentParser(
            add_help=False, parents=[parser])
        suppressed_parser.set_defaults(
            **{k: None for k, v in vars(args).items()})
        args = suppressed_parser.parse_args(input_args)
        return argparse.Namespace(
            **{k: v for k, v in vars(args).items() if v is not None}
        )

    '''
    [GOTO ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY
    IN fairseq/models/__init__.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

    '''
    QUESTION: The following user module related steps are already executed in
    the get_parser step, why is this being done again?

    @readby: rukmangadh.sai@nobroker.in
    '''
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args(input_args)
    utils.import_user_module(usr_args)

    if modify_parser is not None:
        modify_parser(parser)

    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, "arch"):
        model_specific_group = parser.add_argument_group(
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        '''
        [GOTO add_args() IN fariseq/models/wav2vec/wav2vec2_asr.py]
        [  ''      ''       ''      ''      ''    /wav2vec2.py]

        @readby: rukmangadh.sai@nobroker.in
        '''
        ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)

    # Add *-specific args to parser.
    '''
    [GOTO REGISTRIES IN fairseq/registry.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    from fairseq.registry import REGISTRIES

    '''
    Add the flags defined by the additional relevant modules in all the
    registries

    @readby: rukmangadh.sai@nobroker.in
    '''
    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)
    '''
    Add the flags defined by the relevant task in the task registry.

    @readby: rukmangadh.sai@nobroker.in
    '''
    if hasattr(args, "task"):
        from fairseq.tasks import TASK_REGISTRY

        TASK_REGISTRY[args.task].add_args(parser)

    '''
    QUESTION: Do we need to use this?

    @readby: rukmangadh.sai@nobroker.in
    '''
    if getattr(args, "use_bmuf", False):
        # hack to support extra args for block distributed data parallelism
        '''
        [GOTO class FairseqBMUF() IN fairseq/optim/bmuf.py]
        QUESTION: Should we be doing this?

        @readby: rukmangadh.sai@nobroker.in
        '''
        from fairseq.optim.bmuf import FairseqBMUF

        FairseqBMUF.add_args(parser)

    # Modify the parser a second time, since defaults may have been reset
    if modify_parser is not None:
        modify_parser(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None

    # Post-process args.
    if hasattr(args, "max_sentences_valid") and args.max_sentences_valid is None:
        args.max_sentences_valid = args.max_sentences
    if hasattr(args, "max_tokens_valid") and args.max_tokens_valid is None:
        args.max_tokens_valid = args.max_tokens
    if getattr(args, "memory_efficient_fp16", False):
        args.fp16 = True
    if getattr(args, "memory_efficient_bf16", False):
        args.bf16 = True
    args.tpu = getattr(args, "tpu", False)
    args.bf16 = getattr(args, "bf16", False)
    if args.bf16:
        args.tpu = True
    if args.tpu and args.fp16:
        raise ValueError("Cannot combine --fp16 and --tpu, use --bf16 on TPUs")

    if getattr(args, "seed", None) is None:
        args.seed = 1  # default seed for training
        args.no_seed_provided = True
    else:
        args.no_seed_provided = False

    '''
    [GOTO ARCH_CONFIG_REGISTRY IN fairseq/models/__init__.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    # Apply architecture configuration.
    if hasattr(args, "arch"):
        '''
        [GOTO base_architecture() IN fairseq/models/wav2vec/wav2vec2_asr.py]
        The following step sets the wav2vecCtc specific flags to their
        corresponding values.

        @readby: rukmangadh.sai@nobroker.in
        '''
        ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args


'''
FUNCTION: get_parser()

ARGUMENTS:
    desc: NOT USED.
    default_task: The default task for the task registry argument.

[DONE] The following function intiates the argsparse.ArgumentParser object for
parsing arguments provided to a python script. It also defines the command line
arguments common for various modules and also imports the user written modules.

@readby: rukmangadh.sai@nobroker.in
'''


def get_parser(desc, default_task="translation"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args()

    '''
    [GOTO import_user_module() IN fairseq/utils.py]
    Import any user written custom modules like custom tasks, optimzers,
    architectures etc. into this file. The custom modules are registered with
    the fariseq architecture through decorators like @register_task,
    @register_model etc.

    @readby: rukmangadh.sai@nobroker.in
    '''
    utils.import_user_module(usr_args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    # fmt: off
    parser.add_argument('--no-progress-bar',
                        action='store_true', help='disable progress bar')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')
    parser.add_argument('--log-format', default=None, help='log format to use',
                        choices=['json', 'none', 'simple', 'tqdm'])
    '''
    QUESTION: Why should --tensorboard-logdir match --logdir?

    @readby: rukmangadh.sai@nobroker.in
    '''
    parser.add_argument('--tensorboard-logdir', metavar='DIR', default='',
                        help='path to save logs for tensorboard, should match --logdir '
                             'of running tensorboard (default: no tensorboard logging)')
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--cpu', action='store_true',
                        help='use CPU instead of CUDA')

    '''
    QUESTION: Are these tpu, bf16, fp16, memory-efficient-bf16,
    memory-efficient-fp16, fp16-no-flatten-grad, fp16-init-scale,
    fp16-scale-window, fp16-scale-tolerance, min-loss-scale,
    threshold-loss-scale, user-dir, empty-cache-freq, all-gather-list-size,
    model-parallel-size, checkpoint-suffix, quantization-config-path, profile
    options relevant to us?

    @readby: rukmangadh.sai@nobroker.in
    '''
    parser.add_argument('--tpu', action='store_true',
                        help='use TPU instead of CUDA')
    parser.add_argument('--bf16', action='store_true',
                        help='use bfloat16; implies --tpu')
    parser.add_argument('--fp16', action='store_true', help='use FP16')
    parser.add_argument('--memory-efficient-bf16', action='store_true',
                        help='use a memory-efficient version of BF16 training; implies --bf16')
    parser.add_argument('--memory-efficient-fp16', action='store_true',
                        help='use a memory-efficient version of FP16 training; implies --fp16')
    parser.add_argument('--fp16-no-flatten-grads', action='store_true',
                        help='don\'t flatten FP16 grads tensor')
    parser.add_argument('--fp16-init-scale', default=2 ** 7, type=int,
                        help='default FP16 loss scale')
    parser.add_argument('--fp16-scale-window', type=int,
                        help='number of updates before increasing loss scale')
    parser.add_argument('--fp16-scale-tolerance', default=0.0, type=float,
                        help='pct of updates that can overflow before decreasing the loss scale')
    parser.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
                        help='minimum FP16 loss scale, after which training is stopped')
    parser.add_argument('--threshold-loss-scale', type=float,
                        help='threshold FP16 loss scale from below')
    parser.add_argument('--user-dir', default=None,
                        help='path to a python module containing custom extensions (tasks and/or architectures)')
    parser.add_argument('--empty-cache-freq', default=0, type=int,
                        help='how often to clear the PyTorch CUDA cache (0 to disable)')
    parser.add_argument('--all-gather-list-size', default=16384, type=int,
                        help='number of bytes reserved for gathering stats from workers')
    parser.add_argument('--model-parallel-size', type=int, metavar='N',
                        default=1,
                        help='total number of GPUs to parallelize model over')
    parser.add_argument('--checkpoint-suffix', default='',
                        help='suffix to add to the checkpoint file name')
    parser.add_argument('--quantization-config-path', default=None,
                        help='path to quantization config file')
    parser.add_argument('--profile', action='store_true',
                        help='enable autograd profiler emit_nvtx')

    '''
    [GOTO fariseq/registry.py]
    REGISTRIES contains all the registers maintained. The for loop below adds
    the names of names of each register (register_name) as the argument name
    and the default specified while setting up the registry [GOTO
    setup_registry IN fairseq/registry.py] as the default for the argument. The
    choices argument is list of names all the objects registered under that
    particular registry.

    QUESTION: I donot think these registries are required for us but I need to
    be sure about this. So, are these surely unimportant?

    @readby: rukmangadh.sai@nobroker.in
    '''
    from fairseq.registry import REGISTRIES
    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            '--' + registry_name.replace('_', '-'),
            default=REGISTRY['default'],
            choices=REGISTRY['registry'].keys(),
        )

    '''
    [GOTO fariseq/tasks/__init__.py]
    Similar to what happens above. The only differnce seems that TASK_REGISTRY
    seems to be not included in the REGISTRIES variable above.

    QUESTION: I donot think this registry is required for us but I need to be
    sure about this. So, is this surely unimportant?

    @readby: rukmangadh.sai@nobroker.in
    '''
    # Task definitions can be found under fairseq/tasks/
    from fairseq.tasks import TASK_REGISTRY
    parser.add_argument('--task', metavar='TASK', default=default_task,
                        choices=TASK_REGISTRY.keys(),
                        help='task')
    # fmt: on
    return parser


def add_preprocess_args(parser):
    group = parser.add_argument_group("Preprocessing")
    # fmt: off
    group.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                       help="source language")
    group.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                       help="target language")
    group.add_argument("--trainpref", metavar="FP", default=None,
                       help="train file prefix")
    group.add_argument("--validpref", metavar="FP", default=None,
                       help="comma separated, valid file prefixes")
    group.add_argument("--testpref", metavar="FP", default=None,
                       help="comma separated, test file prefixes")
    group.add_argument("--align-suffix", metavar="FP", default=None,
                       help="alignment file suffix")
    group.add_argument("--destdir", metavar="DIR", default="data-bin",
                       help="destination dir")
    group.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--tgtdict", metavar="FP",
                       help="reuse given target dictionary")
    group.add_argument("--srcdict", metavar="FP",
                       help="reuse given source dictionary")
    group.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
                       help="number of target words to retain")
    group.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
                       help="number of source words to retain")
    group.add_argument("--alignfile", metavar="ALIGN", default=None,
                       help="an alignment file (optional)")
    parser.add_argument('--dataset-impl', metavar='FORMAT', default='mmap',
                        choices=get_available_dataset_impl(),
                        help='output dataset implementation')
    group.add_argument("--joined-dictionary", action="store_true",
                       help="Generate joined dictionary")
    group.add_argument("--only-source", action="store_true",
                       help="Only process the source language")
    group.add_argument("--padding-factor", metavar="N", default=8, type=int,
                       help="Pad dictionary size to be multiple of N")
    group.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")
    # fmt: on
    return parser


'''
FUNCTION: add_dataset_args()

ARGUMENTS:
    parser: The ArgumentParser object being used to parse command line
    arguments.
    train: Whether training is being done or not
    gen: Whether generation is being done or not

[DONE] This function adds the arguments required for handling datasets.

@readby: rukmangadh.sai@nobroker.in
'''


def add_dataset_args(parser, train=False, gen=False):
    '''
    By default, ArgumentParser goups command line arguments as 'positional' or
    'optional' when displaying help messages. If there is a more intelligent
    grouping, one can use the add_argument_group() function.
    Refer: https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument_group

    @readby: rukmangadh.sai@nobroker.in
    '''
    group = parser.add_argument_group("Dataset and data loading")
    # fmt: off
    group.add_argument('--num-workers', default=1, type=int, metavar='N',
                       help='how many subprocesses to use for data loading')

    '''
    QUESTION: Are the flags skip-invalid-size-inputs-valid-test, max-tokens,
    max-sentences, required-batch-size-multiple, dataset-impl

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--skip-invalid-size-inputs-valid-test', action='store_true',
                       help='ignore too long or too short lines in valid and test set')
    group.add_argument('--max-tokens', type=int, metavar='N',
                       help='maximum number of tokens in a batch')
    group.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                       help='maximum number of sentences in a batch')
    group.add_argument('--required-batch-size-multiple', default=8, type=int, metavar='N',
                       help='batch size will either be less than this value, '
                            'or a multiple of this value')

    '''
    [GOTO get_available_dataset_impl() IN fairseq/data/indexed_dataset.py]
    QUESTION: Is this flag relevant to us? Anvesh talked about data formats.
    Need to ask him more about it.

    @readby: rukmangadh.sai@nobroker.in
    '''
    parser.add_argument('--dataset-impl', metavar='FORMAT',
                        choices=get_available_dataset_impl(),
                        help='output dataset implementation')
    group.add_argument('--data-buffer-size', default=10, type=int, metavar='N',
                       help='number of batches to preload')
    if train:
        group.add_argument('--train-subset', default='train', metavar='SPLIT',
                           help='data subset to use for training (e.g. train, valid, test)')
        group.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                           help='comma separated list of data subsets to use for validation'
                                ' (e.g. train, valid, test)')
        group.add_argument('--validate-interval', type=int, default=1, metavar='N',
                           help='validate every N epochs')
        group.add_argument('--validate-interval-updates', type=int, default=0, metavar='N',
                           help='validate every N updates')
        group.add_argument('--validate-after-updates', type=int, default=0, metavar='N',
                           help='dont validate until reaching this many updates')
        group.add_argument('--fixed-validation-seed', default=None, type=int, metavar='N',
                           help='specified random seed for validation')
        group.add_argument('--disable-validation', action='store_true',
                           help='disable validation')
        '''
        QUESTION: Do we need max-tokens-valid, max-sentences-valid flags?

        @readby: rukmangadh.sai@nobroker.in
        '''
        group.add_argument('--max-tokens-valid', type=int, metavar='N',
                           help='maximum number of tokens in a validation batch'  # noqa: E501
                                ' (defaults to --max-tokens)')
        group.add_argument('--max-sentences-valid', type=int, metavar='N',
                           help='maximum number of sentences in a validation batch'
                                ' (defaults to --max-sentences)')
        '''
        I think the following flag is for curriculum learning.

        QUESTION: Do we need curriculum flag?

        @readby: rukmangadh.sai@nobroker.in
        '''
        group.add_argument('--curriculum', default=0, type=int, metavar='N',
                           help='don\'t shuffle batches for first N epochs')

    '''
    QUESTION: Do we need generation flags?

    @readby: rukmangadh.sai@nobroker.in
    '''
    if gen:
        group.add_argument('--gen-subset', default='test', metavar='SPLIT',
                           help='data subset to generate (train, valid, test)')
        group.add_argument('--num-shards', default=1, type=int, metavar='N',
                           help='shard generation over N shards')
        group.add_argument('--shard-id', default=0, type=int, metavar='ID',
                           help='id of the shard to generate (id < num_shards)')
    # fmt: on
    return group


'''
FUNCTION: add_distributed_training_args()

[NOTDONE]

@readby: rukmangadh.sai@nobroker.in
'''


def add_distributed_training_args(parser, default_world_size=None):
    group = parser.add_argument_group("Distributed training")
    # fmt: off
    if default_world_size is None:
        default_world_size = max(1, torch.cuda.device_count())
    group.add_argument('--distributed-world-size', type=int, metavar='N',
                       default=default_world_size,
                       help='total number of GPUs across all nodes (default: all visible GPUs)')
    group.add_argument('--distributed-rank', default=0, type=int,
                       help='rank of the current worker')
    group.add_argument('--distributed-backend', default='nccl', type=str,
                       help='distributed backend')
    group.add_argument('--distributed-init-method', default=None, type=str,
                       help='typically tcp://hostname:port that will be used to '
                            'establish initial connetion')
    group.add_argument('--distributed-port', default=-1, type=int,
                       help='port number (not required if using --distributed-init-method)')
    group.add_argument('--device-id', '--local_rank', default=0, type=int,
                       help='which GPU to use (usually configured automatically)')
    group.add_argument('--distributed-no-spawn', action='store_true',
                       help='do not spawn multiple processes even if multiple GPUs are visible')
    # "c10d" is PyTorch's DDP implementation and provides the fastest
    # training. "no_c10d" is a more robust, but slightly slower DDP
    # implementation. Try this if you get warning messages about
    # inconsistent gradients between workers, or if some of your model
    # parameters are not always used.
    group.add_argument('--ddp-backend', default='c10d', type=str,
                       choices=['c10d', 'no_c10d'],
                       help='DistributedDataParallel backend')
    group.add_argument('--bucket-cap-mb', default=25, type=int, metavar='MB',
                       help='bucket size for reduction')
    group.add_argument('--fix-batches-to-gpus', action='store_true',
                       help='don\'t shuffle batches between GPUs; this reduces overall '
                            'randomness and may affect precision but avoids the cost of '
                            're-reading the data')
    group.add_argument('--find-unused-parameters', default=False, action='store_true',
                       help='disable unused parameter detection (not applicable to '
                       'no_c10d ddp-backend')
    group.add_argument('--fast-stat-sync', default=False, action='store_true',
                       help='[deprecated] this is now defined per Criterion')
    group.add_argument('--broadcast-buffers', default=False, action='store_true',
                       help='Copy non-trainable parameters between GPUs, such as '
                       'batchnorm population statistics')

    group.add_argument('--distributed-wrapper', default='DDP', type=str,
                       choices=['DDP', 'SlowMo'],
                       help='DistributedDataParallel backend')
    # Add arguments for SlowMo - these will be used when SlowMo is enabled via above
    group.add_argument('--slowmo-momentum', default=None, type=float,
                       help='SlowMo momentum term; by default use 0.0 for 16 GPUs, '
                            '0.2 for 32 GPUs; 0.5 for 64 GPUs, 0.6 for > 64 GPUs')
    group.add_argument('--slowmo-algorithm', default='LocalSGD', choices=['LocalSGD', 'SGP'],
                       help='whether to use LocalSGD or SGP')
    group.add_argument('--localsgd-frequency', default=3, type=int,
                       help='Local SGD allreduce frequency')
    group.add_argument('--nprocs-per-node', type=int, metavar='N',
                       default=max(1, torch.cuda.device_count()),
                       help='number of GPUs in each node. An allreduce operation across GPUs in '
                            'a node is very fast. Hence, we do allreduce across GPUs in a node, '
                            'and gossip across different nodes')
    # Add argument for ZeRO sharding of OptimizerState(os), gradients(g) and parameters(p)
    group.add_argument('--zero-sharding', default='none', type=str,
                       choices=['none', 'os'],
                       help='ZeRO sharding')
    # fmt: on
    return group


'''
FUNCTION: add_optimization_args()

ARGUMENTS:
    parser: The ArgumentParser object being used for parsing command line
    arguments.

[DONE] This function add the general flags related to optimization to parser.

@readby: rukmangadh.sai@nobroker.in
'''


def add_optimization_args(parser):
    group = parser.add_argument_group("Optimization")
    # fmt: off
    '''
    QUESTION: What happens when the flags --max-epoch and --max-update are left
    to be their default value 0?

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--max-update', '--mu', default=0, type=int, metavar='N',
                       help='force stop training at specified update')
    group.add_argument('--stop-time-hours', default=0, type=float, metavar='N',
                       help='force stop training after specified cumulative time (if >0)')
    '''
    QUESTION: Do we need the flags --clip-norm and --sentence-avg?

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--clip-norm', default=0.0, type=float, metavar='NORM',
                       help='clip threshold of gradients')
    group.add_argument('--sentence-avg', action='store_true',
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of tokens)')
    '''
    QUESTION: What happens when we go beyond the K'th epoch?

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--update-freq', default='1', metavar='N1,N2,...,N_K',
                       type=lambda uf: eval_str_list(uf, type=int),
                       help='update parameters every N_i batches, when in epoch i')
    '''
    QUESTION: What does the 'note' in the following flag's help mean?

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--lr', '--learning-rate', default='0.25', type=eval_str_list,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--min-lr', default=-1, type=float, metavar='LR',
                       help='stop training when the learning rate reaches this minimum')
    '''
    QUESTION: Do we need the --use-bmuf flag?

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--use-bmuf', default=False, action='store_true',
                       help='specify global optimizer for syncing models on different GPUs/shards')
    # fmt: on
    return group


'''
FUNCTION: add_checkpoint_args()

ARGUMENTS:
    parser: The ArgumentParser object used for parsing command line arguments

[DONE] This function adds the checkpointing flags to the parser.

@readby: rukmangadh.sai@nobroker.in
'''


def add_checkpoint_args(parser):
    group = parser.add_argument_group("Checkpointing")
    # fmt: off
    group.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')
    '''
    QUESTION: Will this --restore-file be always checked in the <save-dir>?

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--restore-file', default='checkpoint_last.pt',
                       help='filename from which to load checkpoint '
                            '(default: <save-dir>/checkpoint_last.pt')
    '''
    QUESTION: What does 'meters and lr scheduler will be reset' mean?

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--finetune-from-model', default=None, type=str,
                       help='finetune from a pretrained model; '
                            'note that meters and lr scheduler will be reset')
    group.add_argument('--reset-dataloader', action='store_true',
                       help='if set, does not reload dataloader state from the checkpoint')
    group.add_argument('--reset-lr-scheduler', action='store_true',
                       help='if set, does not load lr scheduler state from the checkpoint')
    group.add_argument('--reset-meters', action='store_true',
                       help='if set, does not load meters from the checkpoint')
    group.add_argument('--reset-optimizer', action='store_true',
                       help='if set, does not load optimizer state from the checkpoint')
    group.add_argument('--optimizer-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override optimizer args when loading a checkpoint')
    group.add_argument('--save-interval', type=int, default=1, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('--save-interval-updates', type=int, default=0, metavar='N',
                       help='save a checkpoint (and validate) every N updates')
    group.add_argument('--keep-interval-updates', type=int, default=-1, metavar='N',
                       help='keep the last N checkpoints saved with --save-interval-updates')
    group.add_argument('--keep-last-epochs', type=int, default=-1, metavar='N',
                       help='keep last N epoch checkpoints')
    group.add_argument('--keep-best-checkpoints', type=int, default=-1, metavar='N',
                       help='keep best N checkpoints based on scores')
    group.add_argument('--no-save', action='store_true',
                       help='don\'t save models or checkpoints')
    group.add_argument('--no-epoch-checkpoints', action='store_true',
                       help='only store last and best checkpoints')
    group.add_argument('--no-last-checkpoints', action='store_true',
                       help='don\'t store last checkpoints')
    group.add_argument('--no-save-optimizer-state', action='store_true',
                       help='don\'t save optimizer-state as part of checkpoint')
    group.add_argument('--best-checkpoint-metric', type=str, default='loss',
                       help='metric to use for saving "best" checkpoints')
    group.add_argument('--maximize-best-checkpoint-metric', action='store_true',
                       help='select the largest metric value for saving "best" checkpoints')
    '''
    QUESTION: How is --patience influenced by --validate-interval?

    @readby: rukmangadh.sai@nobroker.in
    '''
    group.add_argument('--patience', type=int, default=-1, metavar='N',
                       help=('early stop training if valid performance doesn\'t '
                             'improve for N consecutive validation runs; note '
                             'that this is influenced by --validate-interval'))
    # fmt: on
    return group


def add_common_eval_args(group):
    # fmt: off
    group.add_argument('--path', metavar='FILE',
                       help='path(s) to model file(s), colon separated')
    group.add_argument('--remove-bpe', '--post-process', nargs='?', const='@@ ', default=None,
                       help='remove BPE tokens before scoring (can be set to sentencepiece)')
    group.add_argument('--quiet', action='store_true',
                       help='only print final scores')
    group.add_argument('--model-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override model args at generation '
                            'that were used during model training')
    group.add_argument('--results-path', metavar='RESDIR', type=str, default=None,
                       help='path to save eval results (optional)"')
    # fmt: on


def add_eval_lm_args(parser):
    group = parser.add_argument_group("LM Evaluation")
    add_common_eval_args(group)
    # fmt: off
    group.add_argument('--output-word-probs', action='store_true',
                       help='if set, outputs words and their predicted log probabilities to standard output')
    group.add_argument('--output-word-stats', action='store_true',
                       help='if set, outputs word statistics such as word count, average probability, etc')
    group.add_argument('--context-window', default=0, type=int, metavar='N',
                       help='ensures that every evaluated token has access to a context of at least this size,'
                            ' if possible')
    group.add_argument('--softmax-batch', default=sys.maxsize, type=int, metavar='N',
                       help='if BxT is more than this, will batch the softmax over vocab to this amount of tokens'
                            ' in order to fit into GPU memory')
    # fmt: on


def add_generation_args(parser):
    group = parser.add_argument_group("Generation")
    add_common_eval_args(group)
    # fmt: off
    group.add_argument('--beam', default=5, type=int, metavar='N',
                       help='beam size')
    group.add_argument('--nbest', default=1, type=int, metavar='N',
                       help='number of hypotheses to output')
    group.add_argument('--max-len-a', default=0, type=float, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
    group.add_argument('--max-len-b', default=200, type=int, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
    group.add_argument('--min-len', default=1, type=float, metavar='N',
                       help=('minimum generation length'))
    group.add_argument('--match-source-len', default=False, action='store_true',
                       help=('generations should match the source length'))
    group.add_argument('--no-early-stop', action='store_true',
                       help='deprecated')
    group.add_argument('--unnormalized', action='store_true',
                       help='compare unnormalized hypothesis scores')
    group.add_argument('--no-beamable-mm', action='store_true',
                       help='don\'t use BeamableMM in attention layers')
    group.add_argument('--lenpen', default=1, type=float,
                       help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    group.add_argument('--unkpen', default=0, type=float,
                       help='unknown word penalty: <0 produces more unks, >0 produces fewer')
    group.add_argument('--replace-unk', nargs='?', const=True, default=None,
                       help='perform unknown replacement (optionally with alignment dictionary)')
    group.add_argument('--sacrebleu', action='store_true',
                       help='score with sacrebleu')
    group.add_argument('--score-reference', action='store_true',
                       help='just score the reference translation')
    group.add_argument('--prefix-size', default=0, type=int, metavar='PS',
                       help='initialize generation by target prefix of given length')
    group.add_argument('--no-repeat-ngram-size', default=0, type=int, metavar='N',
                       help='ngram blocking such that this size ngram cannot be repeated in the generation')
    group.add_argument('--sampling', action='store_true',
                       help='sample hypotheses instead of using beam search')
    group.add_argument('--sampling-topk', default=-1, type=int, metavar='PS',
                       help='sample from top K likely next words instead of all words')
    group.add_argument('--sampling-topp', default=-1.0, type=float, metavar='PS',
                       help='sample from the smallest set whose cumulative probability mass exceeds p for next words')
    group.add_argument('--constraints', const="ordered", nargs="?", choices=["ordered", "unordered"],
                       help='enables lexically constrained decoding')
    group.add_argument('--temperature', default=1., type=float, metavar='N',
                       help='temperature for generation')
    group.add_argument('--diverse-beam-groups', default=-1, type=int, metavar='N',
                       help='number of groups for Diverse Beam Search')
    group.add_argument('--diverse-beam-strength', default=0.5, type=float, metavar='N',
                       help='strength of diversity penalty for Diverse Beam Search')
    group.add_argument('--diversity-rate', default=-1.0, type=float, metavar='N',
                       help='strength of diversity penalty for Diverse Siblings Search')
    group.add_argument('--print-alignment', action='store_true',
                       help='if set, uses attention feedback to compute and print alignment to source tokens')
    group.add_argument('--print-step', action='store_true')

    # arguments for iterative refinement generator
    group.add_argument('--iter-decode-eos-penalty', default=0.0, type=float, metavar='N',
                       help='if > 0.0, it penalized early-stopping in decoding.')
    group.add_argument('--iter-decode-max-iter', default=10, type=int, metavar='N',
                       help='maximum iterations for iterative refinement.')
    group.add_argument('--iter-decode-force-max-iter', action='store_true',
                       help='if set, run exact the maximum number of iterations without early stop')
    group.add_argument('--iter-decode-with-beam', default=1, type=int, metavar='N',
                       help='if > 1, model will generate translations varying by the lengths.')
    group.add_argument('--iter-decode-with-external-reranker', action='store_true',
                       help='if set, the last checkpoint are assumed to be a reranker to rescore the translations'),
    group.add_argument('--retain-iter-history', action='store_true',
                       help='if set, decoding returns the whole history of iterative refinement')
    group.add_argument('--retain-dropout', action='store_true',
                       help='Use dropout at inference time')
    group.add_argument('--retain-dropout-modules', default=None, nargs='+', type=str,
                       help='if set, only retain dropout for the specified modules; '
                            'if not set, then dropout will be retained for all modules')

    # special decoding format for advanced decoding.
    group.add_argument('--decoding-format', default=None, type=str,
                       choices=['unigram', 'ensemble', 'vote', 'dp', 'bs'])
    # fmt: on
    return group


def add_interactive_args(parser):
    group = parser.add_argument_group("Interactive")
    # fmt: off
    group.add_argument('--buffer-size', default=0, type=int, metavar='N',
                       help='read this many sentences into a buffer before processing them')
    group.add_argument('--input', default='-', type=str, metavar='FILE',
                       help='file to read from; use - for stdin')
    # fmt: on


'''
FUNCTION: add_model_args()

ARGUMENTS:
    parser: The ArgumentParser object being used for parsing command line
    arguments.

[DONE] This function add general model related flags to the parser.

@readby: rukmangadh.sai@nobroker.in
'''


def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off

    # Model definitions can be found under fairseq/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    '''
    [GOTO ARCH_MODEL_REGISTRY IN fariseq/models/__init__.py]

    @readby: rukmangadh.sai@nobroker.in
    '''
    from fairseq.models import ARCH_MODEL_REGISTRY
    group.add_argument('--arch', '-a', metavar='ARCH',
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='model architecture')
    # fmt: on
    return group
