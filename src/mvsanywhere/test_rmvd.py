"""
    Test script for evaluating MVSAnywhere on the Robust Multi-View Depth Benchmark

"""


import os
import torch

import mvsanywhere.options as options
from mvsanywhere.experiment_modules.rmvd_mvsa import MVSA_Wrapped

import rmvd


def main(opts):
    # get dataset
    assert opts.datasets is None, f"Expected zero datasets but got {(opts.datasets)}"

    # path where results for this model
    results_path = os.path.join(
        opts.output_base_path, opts.name, 'rmvd'
    )

    # Set up model. Note that we're not passing in opts as an argument, although
    # we could. We're being pretty stubborn with using the options the model had
    # used when training, saved internally as part of hparams in the checkpoint.
    # You can change this at inference by passing in 'opts=opts,' but there
    # be dragons if you're not careful.
    model = MVSA_Wrapped(opts, use_refinement=True)
    model = rmvd.prepare_custom_model(model)
    
    evaluation = rmvd.create_evaluation(
        evaluation_type="robustmvd",
        out_dir=results_path,
        inputs=["intrinsics", "poses"],
        eval_uncertainty=False,
        max_source_views=7,
        alignment=None,
    )
    results = evaluation(
        model=model,
        eth3d_size=(480, 640),
        kitti_size=(480, 1280),
        dtu_size=(480, 640),
        scannet_size=(480, 640),
        tanks_and_temples_size=(480, 640)
    )


if __name__ == "__main__":
    # don't need grad for test.
    torch.set_grad_enabled(False)

    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)